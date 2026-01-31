import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from mGPT.archs.mgpt_vq import VQVae
from mGPT.utils.human_models import smpl_x, get_coord
from mGPT.utils.render_utils import render_video_from_meshes

# VQVae architecture configs (from configs/vq/re96.yaml and configs/vq/hand192.yaml)
BODY_VAE_CFG = dict(
    quantizer='ema_reset', code_num=96, code_dim=512,
    output_emb_width=512, down_t=2, stride_t=2,
    width=512, depth=3, dilation_growth_rate=3,
    norm=None, activation='relu', nfeats=43,
)
HAND_VAE_CFG = dict(
    quantizer='ema_reset', code_num=192, code_dim=512,
    output_emb_width=512, down_t=2, stride_t=2,
    width=512, depth=3, dilation_growth_rate=3,
    norm=None, activation='relu', nfeats=45,
)


def load_vaes(ckpt_path, device):
    """Load body, left-hand, and right-hand VQVae models from a checkpoint."""
    body_vae = VQVae(**BODY_VAE_CFG).to(device)
    hand_vae = VQVae(**HAND_VAE_CFG).to(device)
    rhand_vae = VQVae(**HAND_VAE_CFG).to(device)

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

    body_dict = OrderedDict()
    hand_dict = OrderedDict()
    rhand_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("rhand_vae."):
            rhand_dict[k.replace("rhand_vae.", "")] = v
        elif k.startswith("hand_vae."):
            hand_dict[k.replace("hand_vae.", "")] = v
        elif k.startswith("motion_vae."):
            body_dict[k.replace("motion_vae.", "")] = v
        elif k.startswith("vae."):
            body_dict[k.replace("vae.", "")] = v

    body_vae.load_state_dict(body_dict, strict=True)
    hand_vae.load_state_dict(hand_dict, strict=True)
    rhand_vae.load_state_dict(rhand_dict, strict=True)

    body_vae.eval()
    hand_vae.eval()
    rhand_vae.eval()
    return body_vae, hand_vae, rhand_vae


def load_mean_std(mean_path, std_path, device):
    """Load and preprocess mean/std for denormalization (skip root + lower body, rearrange expression)."""
    mean = torch.load(mean_path, map_location=device)
    std = torch.load(std_path, map_location=device)
    # Skip root (3) + lower-body joints (3*11 = 33) = first 36 dims
    mean = mean[(3 + 3 * 11):]
    mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
    std = std[(3 + 3 * 11):]
    std = torch.cat([std[:-20], std[-10:]], dim=0)
    return mean, std


def feats2joints(features, mean, std):
    """Convert normalized 133-dim motion features to SMPL-X vertices via forward kinematics."""
    B, T, D = features.shape

    if mean.shape[0] > D:
        mean = mean[:D]
        std = std[:D]

    features = features * std + mean

    if features.shape[-1] == 123:
        features = torch.cat([features, torch.zeros(B, T, 10).to(features)], dim=-1)

    zero_pose = torch.zeros(*features.shape[:-1], 36).to(features)
    shape_param = torch.tensor([[[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
                                   0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]]]).to(features)
    shape_param = shape_param.repeat(B, T, 1).view(B * T, -1)

    # Prepend 36-dim zero root/lower-body, total = 36 + 133 = 169
    features = torch.cat([zero_pose, features], dim=-1).view(B * T, -1)

    vertices, joints = get_coord(
        root_pose=features[..., 0:3],
        body_pose=features[..., 3:66],
        lhand_pose=features[..., 66:111],
        rhand_pose=features[..., 111:156],
        jaw_pose=features[..., 156:159],
        shape=shape_param,
        expr=features[..., 159:169],
    )
    return vertices


def decode_word(word, word2code, body_vae, hand_vae, rhand_vae, mean, std, device):
    """Decode a single word's codebook tokens through the three VQVae decoders into SMPL-X vertices."""
    entry = word2code[word]
    body_tokens = torch.tensor(entry["body"], dtype=torch.long, device=device)
    lhand_tokens = torch.tensor(entry["lhand"], dtype=torch.long, device=device)
    rhand_tokens = torch.tensor(entry["rhand"], dtype=torch.long, device=device)

    body_tokens = torch.clamp(body_tokens, 0, body_vae.code_num - 1)
    lhand_tokens = torch.clamp(lhand_tokens, 0, hand_vae.code_num - 1)
    rhand_tokens = torch.clamp(rhand_tokens, 0, rhand_vae.code_num - 1)

    with torch.no_grad():
        body_motion = body_vae.decode(body_tokens)      # (1, T_body*4, 43)
        lhand_motion = hand_vae.decode(lhand_tokens)     # (1, T_hand*4, 45)
        rhand_motion = rhand_vae.decode(rhand_tokens)    # (1, T_rhand*4, 45)

    # Pad shorter sequences to the longest
    max_len = max(body_motion.shape[1], lhand_motion.shape[1], rhand_motion.shape[1])
    if body_motion.shape[1] < max_len:
        body_motion = F.pad(body_motion, (0, 0, 0, max_len - body_motion.shape[1]), mode='replicate')
    if lhand_motion.shape[1] < max_len:
        lhand_motion = F.pad(lhand_motion, (0, 0, 0, max_len - lhand_motion.shape[1]), mode='replicate')
    if rhand_motion.shape[1] < max_len:
        rhand_motion = F.pad(rhand_motion, (0, 0, 0, max_len - rhand_motion.shape[1]), mode='replicate')

    # Assemble 133-dim feature vector:
    #   [0:30]    body pose
    #   [30:75]   left hand (45 dims)
    #   [75:120]  right hand (45 dims)
    #   [120:133] face (13 dims, from body VAE output dims 30:43)
    feats = torch.zeros(1, max_len, 133, device=device)
    feats[0, :, :30] = body_motion[0, :, :30]
    feats[0, :, 120:133] = body_motion[0, :, 30:43]
    feats[0, :, 30:75] = lhand_motion[0]
    feats[0, :, 75:120] = rhand_motion[0]

    vertices = feats2joints(feats, mean, std)
    return vertices.cpu().numpy()  # (T, V, 3)


def main():
    parser = argparse.ArgumentParser(description="Decode word tokens from word2code.json into MP4 motion videos.")
    parser.add_argument("--words", nargs="+", required=True,
                        help="Words to decode (must exist as keys in word2code.json).")
    parser.add_argument("--word2code_path", type=str, default="scripts/word2code.json",
                        help="Path to word2code.json.")
    parser.add_argument("--ckpt_path", type=str, default="../pretrained/tokenizer.ckpt",
                        help="Path to the VAE tokenizer checkpoint.")
    parser.add_argument("--mean_path", type=str, default="../data/CSL-Daily/mean.pt",
                        help="Path to mean.pt for denormalization.")
    parser.add_argument("--std_path", type=str, default="../data/CSL-Daily/std.pt",
                        help="Path to std.pt for denormalization.")
    parser.add_argument("--output_dir", type=str, default="./results/output_word_videos",
                        help="Directory to save output MP4 files.")
    parser.add_argument("--fps", type=int, default=20, help="Video frames per second.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load word-to-code mapping
    print(f"Loading word2code from {args.word2code_path} ...")
    with open(args.word2code_path, "r") as f:
        word2code = json.load(f)

    # Validate all requested words exist
    missing = [w for w in args.words if w not in word2code]
    if missing:
        available = list(word2code.keys())[:30]
        print(f"Error: words not found in word2code.json: {missing}")
        print(f"Available words (first 30): {available}")
        return

    # Load VQVae models
    print(f"Loading VQVae models from {args.ckpt_path} ...")
    body_vae, hand_vae, rhand_vae = load_vaes(args.ckpt_path, device)

    # Load normalisation statistics
    print(f"Loading mean/std from {args.mean_path}, {args.std_path} ...")
    mean, std = load_mean_std(args.mean_path, args.std_path, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Decode and render each word
    for word in args.words:
        print(f"\nDecoding '{word}' ({len(word2code[word]['body'])} body tokens, "
              f"{len(word2code[word]['lhand'])} lhand tokens, "
              f"{len(word2code[word]['rhand'])} rhand tokens) ...")
        vertices = decode_word(word, word2code, body_vae, hand_vae, rhand_vae, mean, std, device)

        save_path = os.path.join(args.output_dir, f"{word}.mp4")
        render_video_from_meshes(
            verts_list=vertices,
            faces=smpl_x.face,
            save_path=save_path,
            fps=args.fps,
        )
        print(f"Saved: {save_path}")

    print(f"\nDone. {len(args.words)} video(s) saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
