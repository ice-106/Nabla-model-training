"""
pkl2mp4_mano.py
---------------
Converts SMPL-X feature .pkl sequences to .mp4 videos, with optional
MANO hand replacement for improved hand mesh quality.

Extends the original pkl2mp4 pipeline with:
  --mano_left   MANO_LEFT.pkl
  --mano_right  MANO_RIGHT.pkl
  --vertex_ids  MANO_SMPLX_vertex_ids.pkl

When all three are provided, the SMPLX body mesh is rendered as normal, but
the 778 hand vertices on each side are replaced per-frame with MANO-rendered
hands, anchored to the SMPLX body wrist joints (joints 20 / 21). This removes
the hand shape artifacts that appear when SMPLX drives hands via its internal
PCA/mean prior.

Coordinate alignment (wrist anchoring)
---------------------------------------
The MANO forward pass runs with transl=0 to keep everything in MANO model
space. After the pass, the wrist offset is computed as:

    offset_L = smplx_joints[20] - mano_L_joints[0]
    offset_R = smplx_joints[21] - mano_R_joints[0]

and applied to MANO vertices before transplanting into the SMPLX mesh.
This works correctly regardless of the body's global orientation (upright,
upside-down, etc.) because it uses the actual SMPLX joint positions as
anchors.

Usage
-----
  # Original mode (unchanged)
  python pkl2mp4_mano.py \\
      --input_dir  /path/to/pkls \\
      --output_dir /path/to/out  \\
      --mean_path  mean.pt \\
      --std_path   std.pt

  # With MANO hand replacement
  python pkl2mp4_mano.py \\
      --input_dir  /path/to/pkls \\
      --output_dir /path/to/out  \\
      --mean_path  mean.pt \\
      --std_path   std.pt \\
      --mano_left  MANO_LEFT.pkl \\
      --mano_right MANO_RIGHT.pkl \\
      --vertex_ids MANO_SMPLX_vertex_ids.pkl

All original args (--fps, --rot6d, --num_samples, --seed, --device, --type)
are preserved unchanged.
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import copy
import csv
import gc
import pickle
import random
import warnings

import numpy as np
import torch
from tqdm import tqdm

from mGPT.utils.human_models import smpl_x
from mGPT.utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle
from mGPT.utils.render_utils import render_video_from_meshes

warnings.filterwarnings("ignore")

# SMPLX joint indices for wrist anchoring
SMPLX_LEFT_WRIST  = 20
SMPLX_RIGHT_WRIST = 21

# ---------------------------------------------------------------------------
# SMPL-X layer cache (original)
# ---------------------------------------------------------------------------

_smplx_layer_cache = {}


def get_coord_device(root_pose, body_pose, lhand_pose, rhand_pose,
                     jaw_pose, shape, expr, device):
    """Run SMPL-X forward pass. Returns (vertices, joints)."""
    if device not in _smplx_layer_cache:
        _smplx_layer_cache[device] = copy.deepcopy(smpl_x.layer['neutral']).to(device)
    smplx_layer = _smplx_layer_cache[device]

    batch_size = root_pose.shape[0]
    zero_pose = torch.zeros((batch_size, 3), dtype=torch.float32,
                            device=root_pose.device)

    output = smplx_layer(
        betas=shape,
        body_pose=body_pose,
        global_orient=root_pose,
        right_hand_pose=rhand_pose,
        left_hand_pose=lhand_pose,
        jaw_pose=jaw_pose,
        leye_pose=zero_pose,
        reye_pose=zero_pose,
        expression=expr,
    )
    return output.vertices, output.joints


# ---------------------------------------------------------------------------
# MANO layer cache
# ---------------------------------------------------------------------------

_mano_layer_cache = {}


def get_mano_layer(mano_path: str, is_rhand: bool, device: str):
    """Load and cache a MANO model."""
    import smplx as smplx_lib
    key = (mano_path, is_rhand, device)
    if key not in _mano_layer_cache:
        model = smplx_lib.create(
            model_path=mano_path,
            model_type="mano",
            is_rhand=is_rhand,
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1,
        ).to(device)
        _mano_layer_cache[key] = model
    return _mano_layer_cache[key]


# ---------------------------------------------------------------------------
# Feature -> vertices (original, unchanged)
# ---------------------------------------------------------------------------

def feats2joints(features, mean, std, device, rot6d=False, chunk_size=32):
    """
    Unnormalize features and run SMPL-X forward pass.
    Returns (vertices, joints) both as CPU tensors.
    """
    B, T, D = features.shape

    if mean.shape[0] > D:
        mean = mean[:D]
        std  = std[:D]

    features = features * std + mean

    if features.shape[-1] == 123:
        features = torch.cat(
            [features, torch.zeros(B, T, 10).to(features)], dim=-1)

    zero_pose = torch.zeros(*features.shape[:-1], 36).to(features)
    shape_param_base = torch.tensor(
        [[[-0.07284723,  0.1795129, -0.27608207,  0.135155,   0.10748172,
            0.16037364, -0.01616933, -0.03450319,  0.01369138,  0.01108842]]]
    ).to(features)

    B, T = features.shape[:2]

    if rot6d:
        expr     = features[..., -10:]
        features = features[..., :-10].view(B, T, -1, 6)
        features = matrix_to_axis_angle(rotation_6d_to_matrix(features))
        features = features.view(B, T, -1)
        features = torch.cat([features, expr], dim=-1)

    features    = torch.cat([zero_pose, features], dim=-1).view(B * T, -1)
    shape_param = shape_param_base.repeat(B, T, 1).view(B * T, -1)

    all_vertices, all_joints = [], []
    total_frames = features.shape[0]

    with torch.no_grad():
        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            cf  = features[start:end]
            cs  = shape_param[start:end]

            v, j = get_coord_device(
                root_pose  = cf[..., 0:3],
                body_pose  = cf[..., 3:66],
                lhand_pose = cf[..., 66:111],
                rhand_pose = cf[..., 111:156],
                jaw_pose   = cf[..., 156:159],
                shape      = cs,
                expr       = cf[..., 159:169],
                device     = device,
            )
            all_vertices.append(v.cpu())
            all_joints.append(j.cpu())

    vertices = torch.cat(all_vertices, dim=0)   # (B*T, V, 3)
    joints   = torch.cat(all_joints,   dim=0)   # (B*T, J, 3)
    return vertices, joints


# ---------------------------------------------------------------------------
# MANO hand replacement (new)
# ---------------------------------------------------------------------------

def apply_mano_hands(vertices: np.ndarray,
                     joints:   np.ndarray,
                     features_unnorm: torch.Tensor,
                     mano_left_path:  str,
                     mano_right_path: str,
                     vertex_ids:      dict,
                     device:          str) -> np.ndarray:
    """
    Replace hand vertex regions of SMPLX body meshes with MANO-rendered
    hands, anchored to the SMPLX body wrist joints.

    Feature layout (after unnorm, before zero_pose prepend):
        feats[30:75]  = smplx_lhand_pose  (45-dim axis-angle)
        feats[75:120] = smplx_rhand_pose  (45-dim axis-angle)

    Parameters
    ----------
    vertices        : np.ndarray (T, 10475, 3)   SMPLX body vertices
    joints          : np.ndarray (T, J, 3)        SMPLX joints (20=L wrist, 21=R wrist)
    features_unnorm : torch.Tensor (T, 133)       unnormalized features (CPU)
    mano_left_path  : str
    mano_right_path : str
    vertex_ids      : dict  {'left_hand': (778,), 'right_hand': (778,)}
    device          : str

    Returns
    -------
    np.ndarray (T, 10475, 3)  with hand regions replaced
    """
    mano_l = get_mano_layer(mano_left_path,  is_rhand=False, device=device)
    mano_r = get_mano_layer(mano_right_path, is_rhand=True,  device=device)

    l_ids = vertex_ids["left_hand"]   # (778,)
    r_ids = vertex_ids["right_hand"]  # (778,)

    T = vertices.shape[0]
    out = vertices.copy()

    # Process frame by frame to keep VRAM low
    # (MANO batch_size=1 avoids large allocations per frame)
    for i in range(T):
        # Extract hand poses for this frame (unnormalized features)
        # feats layout (169 total after prepend): lhand=[66:111], rhand=[111:156]
        # In 133-dim: lhand=[30:75], rhand=[75:120]
        lhand_pose = features_unnorm[i, 30:75].unsqueeze(0).to(device)  # (1, 45)
        rhand_pose = features_unnorm[i, 75:120].unsqueeze(0).to(device) # (1, 45)

        # Run MANO (transl=0, orient=0 — only pose matters; we anchor via wrist offset)
        zero3 = torch.zeros(1, 3, device=device)

        with torch.no_grad():
            out_l = mano_l(hand_pose=lhand_pose,  global_orient=zero3, transl=zero3)
            out_r = mano_r(hand_pose=rhand_pose,  global_orient=zero3, transl=zero3)

        l_verts  = out_l.vertices.squeeze(0).cpu().numpy()   # (778, 3)
        l_wrist  = out_l.joints.squeeze(0)[0].cpu().numpy()  # (3,)
        r_verts  = out_r.vertices.squeeze(0).cpu().numpy()
        r_wrist  = out_r.joints.squeeze(0)[0].cpu().numpy()

        # Wrist anchoring: translate MANO hand so its wrist matches SMPLX body wrist
        smplx_l_wrist = joints[i, SMPLX_LEFT_WRIST]    # (3,)
        smplx_r_wrist = joints[i, SMPLX_RIGHT_WRIST]   # (3,)

        offset_l = smplx_l_wrist - l_wrist
        offset_r = smplx_r_wrist - r_wrist

        out[i, l_ids] = l_verts + offset_l
        out[i, r_ids] = r_verts + offset_r

    return out


# ---------------------------------------------------------------------------
# Batch-friendly MANO (optional optimisation for chunk_size > 1)
# ---------------------------------------------------------------------------

def apply_mano_hands_batched(vertices: np.ndarray,
                              joints:   np.ndarray,
                              features_unnorm: torch.Tensor,
                              mano_left_path:  str,
                              mano_right_path: str,
                              vertex_ids:      dict,
                              device:          str,
                              chunk_size:      int = 32) -> np.ndarray:
    """
    Batched version of apply_mano_hands.
    Loads MANO once, processes frames in chunks for throughput.
    """
    import smplx as smplx_lib

    # Load with dynamic batch size (re-create per chunk if needed)
    l_ids = vertex_ids["left_hand"]
    r_ids = vertex_ids["right_hand"]
    T = vertices.shape[0]
    out = vertices.copy()

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        bs  = end - start

        lhand_pose = features_unnorm[start:end, 30:75].to(device)   # (bs, 45)
        rhand_pose = features_unnorm[start:end, 75:120].to(device)  # (bs, 45)

        zero3 = torch.zeros(bs, 3, device=device)

        # Create batched models (smplx doesn't cache batch_size changes well,
        # so we create fresh ones per chunk -- cheap since no weight loading)
        mano_l = smplx_lib.create(
            model_path=mano_left_path,  model_type="mano",
            is_rhand=False, use_pca=False, flat_hand_mean=True, batch_size=bs
        ).to(device)
        mano_r = smplx_lib.create(
            model_path=mano_right_path, model_type="mano",
            is_rhand=True,  use_pca=False, flat_hand_mean=True, batch_size=bs
        ).to(device)

        with torch.no_grad():
            out_l = mano_l(hand_pose=lhand_pose, global_orient=zero3, transl=zero3)
            out_r = mano_r(hand_pose=rhand_pose, global_orient=zero3, transl=zero3)

        l_verts = out_l.vertices.cpu().numpy()           # (bs, 778, 3)
        l_wrist = out_l.joints[:, 0].cpu().numpy()       # (bs, 3)
        r_verts = out_r.vertices.cpu().numpy()
        r_wrist = out_r.joints[:, 0].cpu().numpy()

        smplx_l_wrists = joints[start:end, SMPLX_LEFT_WRIST]   # (bs, 3)
        smplx_r_wrists = joints[start:end, SMPLX_RIGHT_WRIST]  # (bs, 3)

        offsets_l = smplx_l_wrists - l_wrist   # (bs, 3)
        offsets_r = smplx_r_wrists - r_wrist

        # Apply per-frame offsets (broadcast over 778 verts)
        for bi in range(bs):
            fi = start + bi
            out[fi, l_ids] = l_verts[bi] + offsets_l[bi]
            out[fi, r_ids] = r_verts[bi] + offsets_r[bi]

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert SMPL-X PKL sequences to .mp4 videos, "
                    "optionally with MANO hand replacement."
    )

    # --- Original args (unchanged) ---
    parser.add_argument("--input_dir",   type=str, required=True,
                        help="Directory containing .pkl files.")
    parser.add_argument("--output_dir",  type=str, required=True,
                        help="Directory to save output .mp4 files.")
    parser.add_argument("--mean_path",   type=str,
                        default="../data/CSL-Daily/mean.pt",
                        help="Path to mean.pt")
    parser.add_argument("--std_path",    type=str,
                        default="../data/CSL-Daily/std.pt",
                        help="Path to std.pt")
    parser.add_argument("--rot6d",       action="store_true", default=False,
                        help="Whether using 6D rotation.")
    parser.add_argument("--fps",         type=int, default=20,
                        help="Frames per second.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to process. None = all.")
    parser.add_argument("--seed",        type=int, default=1234,
                        help="Random seed for sampling.")
    parser.add_argument("--device",      type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use.")
    parser.add_argument("--type",        type=str, default="result",
                        choices=["result", "reference"],
                        help="Feature key to render: result (feats_rst) or "
                             "reference (feats_ref).")

    # --- New MANO args ---
    parser.add_argument("--mano_dir",   type=str, default=None,
                        help="Directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl. "
                             "If provided, enables MANO hand replacement. "
                             "Overrides --mano_left and --mano_right if set.")
    parser.add_argument("--vertex_ids",  type=str, default=None,
                        help="[MANO] Path to MANO_SMPLX_vertex_ids.pkl.")
    parser.add_argument("--mano_chunk",  type=int, default=32,
                        help="[MANO] Frames per MANO batch (default: 32). "
                             "Reduce if OOM on GPU.")

    args = parser.parse_args()

    # Validate MANO args
    mano_args = [args.mano_dir, args.vertex_ids]
    use_mano  = all(mano_args)
    if any(mano_args) and not use_mano:
        parser.error("--mano_dir, and --vertex_ids must all be "
                     "provided together to enable MANO hand replacement.")

    mano_left  = (str(str(args.mano_dir) + "/MANO_LEFT.pkl")
                  if args.mano_dir else None)
    mano_right = (str(str(args.mano_dir) + "/MANO_RIGHT.pkl")
                  if args.mano_dir else None)
    device = args.device

    # Load mean / std
    if not os.path.exists(args.mean_path) or not os.path.exists(args.std_path):
        print(f"Error: Mean or Std not found at "
              f"{args.mean_path} / {args.std_path}")
        return

    print(f"Loading mean/std ...")
    mean = torch.load(args.mean_path, map_location=device)
    std  = torch.load(args.std_path,  map_location=device)

    mean = mean[(3 + 3 * 11):]
    mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
    std  = std[(3 + 3 * 11):]
    std  = torch.cat([std[:-20],  std[-10:]],  dim=0)

    # Load MANO vertex correspondence
    vertex_ids = None
    if use_mano:
        with open(args.vertex_ids, "rb") as f:
            vertex_ids = pickle.load(f, encoding="latin1")
        print(f"MANO hand replacement enabled.")
        print(f"  MANO_LEFT  : {mano_left}")
        print(f"  MANO_RIGHT : {mano_right}")
        print(f"  vertex_ids : {args.vertex_ids}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect pkl files
    pkl_files = sorted(
        f for f in os.listdir(args.input_dir) if f.endswith(".pkl")
    )

    if args.num_samples is not None:
        random.seed(args.seed)
        random.shuffle(pkl_files)
        pkl_files = pkl_files[:args.num_samples]
        print(f"Using {len(pkl_files)} samples (seed={args.seed}).")

    print(f"Processing {len(pkl_files)} .pkl files ...")

    target_key  = "feats_rst" if args.type == "result" else "feats_ref"
    faces       = smpl_x.face
    metadata    = []

    for pkl_file in tqdm(pkl_files):
        pkl_path   = os.path.join(args.input_dir, pkl_file)
        base_name  = os.path.splitext(pkl_file)[0]
        video_path = os.path.join(args.output_dir, f"{base_name}.mp4")

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            features     = None
            text_content = ""

            if isinstance(data, dict):
                if target_key not in data:
                    tqdm.write(f"Warning: '{target_key}' not in {pkl_file}. "
                               f"Keys: {list(data.keys())}")
                    continue
                features = data[target_key]
                text_content = data.get("text", "")
            else:
                features = data

            if features is None:
                tqdm.write(f"Skipping {pkl_file}: no features.")
                continue

            metadata.append([base_name, text_content])

            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features)

            features = features.to(device).float()
            if features.dim() == 2:
                features = features.unsqueeze(0)          # (1, T, D)

            if features.shape[-1] != mean.shape[-1]:
                tqdm.write(f"Warning: feature dim {features.shape[-1]} != "
                           f"mean dim {mean.shape[-1]}. feats2joints will handle it.")

            # ---- SMPLX forward pass ----
            vertices, joints = feats2joints(
                features, mean, std,
                device=device, rot6d=args.rot6d
            )
            # vertices: (T, 10475, 3)  joints: (T, J, 3)  -- both CPU tensors

            del features
            gc.collect()

            if vertices.shape[0] == 0 or vertices.shape[1] < 100:
                tqdm.write(f"Error: bad vertex shape {vertices.shape} in {pkl_file}")
                continue

            vertices = vertices.numpy()
            joints   = joints.numpy()

            # ---- MANO hand replacement ----
            if use_mano:
                # Recompute unnormalized features for hand pose extraction
                # We need the raw (T, 133) tensor for this file
                with open(pkl_path, "rb") as f:
                    data2 = pickle.load(f)
                raw_feats = data2[target_key] if isinstance(data2, dict) else data2
                if isinstance(raw_feats, np.ndarray):
                    raw_feats = torch.from_numpy(raw_feats)
                raw_feats = raw_feats.float()              # (T, D)
                if raw_feats.dim() == 3:
                    raw_feats = raw_feats.squeeze(0)       # (T, D)

                # Unnormalize to get axis-angle hand params
                D = raw_feats.shape[-1]
                m = mean[:D].cpu()
                s = std[:D].cpu()
                feats_unnorm = raw_feats.cpu() * s + m    # (T, 133)

                vertices = apply_mano_hands_batched(
                    vertices     = vertices,
                    joints       = joints,
                    features_unnorm = feats_unnorm,
                    mano_left_path  = mano_left,
                    mano_right_path = mano_right,
                    vertex_ids      = vertex_ids,
                    device          = device,
                    chunk_size      = args.mano_chunk,
                )
                del feats_unnorm, data2

            # ---- Render ----
            render_video_from_meshes(
                verts_list = vertices,
                faces      = faces,
                save_path  = video_path,
                fps        = args.fps,
            )

        except Exception as e:
            tqdm.write(f"Failed to process {pkl_file}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()

    # Write metadata CSV
    csv_path = os.path.join(args.output_dir, "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Text"])
        writer.writerows(metadata)
    print(f"Metadata written to {csv_path}")


if __name__ == "__main__":
    main()