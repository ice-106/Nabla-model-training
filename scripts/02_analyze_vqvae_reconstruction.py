"""
VQVAE Reconstruction Analysis for Thai Sign Language

Investigates whether the pretrained VQVAE tokenizer (trained on ASL/CSL/DGS)
can reconstruct Thai sign language motions. Measures codebook coverage,
quantization error, and reconstruction quality.

Usage:
    python agentScripts/analyze_vqvae_reconstruction.py \
        --ckpt_path ../pretrained/tokenizer.ckpt \
        --mean_path ../data/CSL-Daily/mean.pt \
        --std_path ../data/CSL-Daily/std.pt \
        --thai_data_dir ../data/Thai \
        --output_dir ./results/vqvae_analysis \
        --render_videos
"""

import os
import sys
import json
import pickle
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mGPT.archs.mgpt_vq import VQVae
from mGPT.utils.human_models import smpl_x, get_coord
from mGPT.utils.render_utils import render_video_from_meshes

# ---------------------------------------------------------------------------
# VAE configs (from configs/vq/re96.yaml and configs/vq/hand192.yaml)
# ---------------------------------------------------------------------------
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

# SMPL-X parameter keys (same as mGPT/data/humanml/load_data.py)
SMPLX_KEYS = [
    'smplx_root_pose',    # 3
    'smplx_body_pose',    # 63
    'smplx_lhand_pose',   # 45
    'smplx_rhand_pose',   # 45
    'smplx_jaw_pose',     # 3
    'smplx_shape',        # 10
    'smplx_expr',         # 10
]


# ---------------------------------------------------------------------------
# Reusable utilities (from decode_word_tokens.py)
# ---------------------------------------------------------------------------
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
    """Load and preprocess mean/std (skip root + lower body, rearrange expression)."""
    mean = torch.load(mean_path, map_location=device, weights_only=False)
    std = torch.load(std_path, map_location=device, weights_only=False)
    # Skip root (3) + lower-body joints (3*11 = 33) = first 36 dims
    mean = mean[(3 + 3 * 11):]
    mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
    std = std[(3 + 3 * 11):]
    std = torch.cat([std[:-20], std[-10:]], dim=0)
    return mean, std


def feats2joints(features, mean, std):
    """Convert normalized 133-dim motion features to SMPL-X vertices."""
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


# ---------------------------------------------------------------------------
# Thai data loading
# ---------------------------------------------------------------------------
def load_thai_poses(data_dir):
    """
    Scan {data_dir}/poses/*/ directories and load Thai motion samples.
    Returns list of (name, features_array[T, 133]) tuples.
    """
    poses_dir = os.path.join(data_dir, 'poses')
    if not os.path.isdir(poses_dir):
        print(f"ERROR: Poses directory not found: {poses_dir}")
        return []

    samples = []
    for name in sorted(os.listdir(poses_dir))[:150]: # Hardcode to load only 150 sample
        sample_dir = os.path.join(poses_dir, name)
        if not os.path.isdir(sample_dir):
            continue

        frame_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.pkl')])
        if len(frame_files) < 4:
            print(f"  Skipping {name}: only {len(frame_files)} frames (need >= 4)")
            continue

        clip_poses = np.zeros([len(frame_files), 179])
        for frame_id, fname in enumerate(frame_files):
            fpath = os.path.join(sample_dir, fname)
            with open(fpath, 'rb') as f:
                poses = pickle.load(f)
            pose = np.concatenate([poses[key] for key in SMPLX_KEYS], axis=0)
            clip_poses[frame_id] = pose

        # Preprocess: skip root(3) + lower body(33) = first 36 dims
        clip_poses = clip_poses[:, (3 + 3 * 11):]
        # Remove shape(10), keep expression(10): remove dims [-20:-10]
        clip_poses = np.concatenate([clip_poses[:, :-20], clip_poses[:, -10:]], axis=1)  # -> 133 dims

        samples.append((name, clip_poses))
        print(f"  Loaded {name}: {clip_poses.shape[0]} frames, {clip_poses.shape[1]} dims")

    return samples


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------
def pad_to_multiple(tensor, multiple=4):
    """Pad temporal dimension to nearest multiple (replicate last frame)."""
    T = tensor.shape[1]
    remainder = T % multiple
    if remainder == 0:
        return tensor, T
    pad_len = multiple - remainder
    tensor = F.pad(tensor, (0, 0, 0, pad_len), mode='replicate')
    return tensor, T


def compute_latent_distances(vae, features_part, device):
    """
    Compute L2 distances from encoder outputs to their nearest codebook entries.
    Returns per-timestep distances in 512-dim latent space.
    """
    with torch.no_grad():
        x_in = vae.preprocess(features_part)       # [1, nfeats, T] -> [1, nfeats, T]
        x_encoder = vae.encoder(x_in)               # [1, 512, T_compressed]
        x_flat = vae.postprocess(x_encoder)          # [1, T_compressed, 512]
        x_flat = x_flat.contiguous().view(-1, x_flat.shape[-1])  # [T_compressed, 512]

        # Find nearest codebook entry
        code_idx = vae.quantizer.quantize(x_flat)    # [T_compressed]
        x_quantized = vae.quantizer.dequantize(code_idx)  # [T_compressed, 512]

        # L2 distance per timestep
        distances = torch.norm(x_flat - x_quantized, dim=-1)  # [T_compressed]

    return distances.cpu().numpy(), code_idx.cpu().numpy()


def analyze_single_sample(name, raw_features, body_vae, hand_vae, rhand_vae, mean, std, device):
    """Run full VQVAE analysis on a single Thai sample."""
    T_orig = raw_features.shape[0]

    # Normalize with CSL-Daily stats
    features_tensor = torch.tensor(raw_features, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, 133]
    features_norm = (features_tensor - mean) / (std + 1e-10)

    # Split into body parts
    pose_re = torch.cat([features_norm[..., :30], features_norm[..., 120:]], dim=-1)  # [1, T, 43]
    pose_lhand = features_norm[..., 30:75]   # [1, T, 45]
    pose_rhand = features_norm[..., 75:120]  # [1, T, 45]

    # Pad to multiple of 4
    pose_re_pad, _ = pad_to_multiple(pose_re)
    pose_lhand_pad, _ = pad_to_multiple(pose_lhand)
    pose_rhand_pad, _ = pad_to_multiple(pose_rhand)

    results = {'name': name, 'num_frames': T_orig}

    with torch.no_grad():
        # --- Forward pass (reconstruction + commit loss + perplexity) ---
        body_recon, body_commit, body_perp = body_vae(pose_re_pad)
        lhand_recon, lhand_commit, lhand_perp = hand_vae(pose_lhand_pad)
        rhand_recon, rhand_commit, rhand_perp = rhand_vae(pose_rhand_pad)

        # Trim back to original length
        body_recon = body_recon[:, :T_orig, :]
        lhand_recon = lhand_recon[:, :T_orig, :]
        rhand_recon = rhand_recon[:, :T_orig, :]

        # --- Encode to get code indices ---
        body_codes, _ = body_vae.encode(pose_re_pad)
        lhand_codes, _ = hand_vae.encode(pose_lhand_pad)
        rhand_codes, _ = rhand_vae.encode(pose_rhand_pad)

    # --- Commitment loss and perplexity ---
    results['commit_loss'] = {
        'body': body_commit.item(),
        'lhand': lhand_commit.item(),
        'rhand': rhand_commit.item(),
    }
    results['perplexity'] = {
        'body': body_perp.item(),
        'lhand': lhand_perp.item(),
        'rhand': rhand_perp.item(),
    }

    # --- Latent space quantization error ---
    body_dists, body_idx = compute_latent_distances(body_vae, pose_re_pad, device)
    lhand_dists, lhand_idx = compute_latent_distances(hand_vae, pose_lhand_pad, device)
    rhand_dists, rhand_idx = compute_latent_distances(rhand_vae, pose_rhand_pad, device)

    results['latent_l2_distance'] = {
        'body': {'mean': float(np.mean(body_dists)), 'median': float(np.median(body_dists)),
                 'max': float(np.max(body_dists)), 'std': float(np.std(body_dists))},
        'lhand': {'mean': float(np.mean(lhand_dists)), 'median': float(np.median(lhand_dists)),
                  'max': float(np.max(lhand_dists)), 'std': float(np.std(lhand_dists))},
        'rhand': {'mean': float(np.mean(rhand_dists)), 'median': float(np.median(rhand_dists)),
                  'max': float(np.max(rhand_dists)), 'std': float(np.std(rhand_dists))},
    }

    # --- Code indices for utilization analysis ---
    results['code_indices'] = {
        'body': body_codes[0].cpu().numpy().tolist(),
        'lhand': lhand_codes[0].cpu().numpy().tolist(),
        'rhand': rhand_codes[0].cpu().numpy().tolist(),
    }

    # --- Reconstruction error in normalized feature space ---
    # Reassemble original normalized 133-dim
    orig_norm = features_norm[:, :T_orig, :]

    # Reassemble reconstructed normalized 133-dim
    recon_norm = torch.zeros_like(orig_norm)
    recon_norm[..., :30] = body_recon[..., :30]
    recon_norm[..., 120:133] = body_recon[..., 30:43]
    recon_norm[..., 30:75] = lhand_recon
    recon_norm[..., 75:120] = rhand_recon

    # Overall
    l1_norm = torch.mean(torch.abs(orig_norm - recon_norm)).item()
    l2_norm = torch.sqrt(torch.mean((orig_norm - recon_norm) ** 2)).item()

    # Per body part
    body_dims = list(range(0, 30)) + list(range(120, 133))
    l1_body = torch.mean(torch.abs(orig_norm[..., body_dims] - recon_norm[..., body_dims])).item()
    l2_body = torch.sqrt(torch.mean((orig_norm[..., body_dims] - recon_norm[..., body_dims]) ** 2)).item()

    l1_lhand = torch.mean(torch.abs(orig_norm[..., 30:75] - recon_norm[..., 30:75])).item()
    l2_lhand = torch.sqrt(torch.mean((orig_norm[..., 30:75] - recon_norm[..., 30:75]) ** 2)).item()

    l1_rhand = torch.mean(torch.abs(orig_norm[..., 75:120] - recon_norm[..., 75:120])).item()
    l2_rhand = torch.sqrt(torch.mean((orig_norm[..., 75:120] - recon_norm[..., 75:120]) ** 2)).item()

    results['reconstruction_error_normalized'] = {
        'overall': {'l1': l1_norm, 'l2_rmse': l2_norm},
        'body': {'l1': l1_body, 'l2_rmse': l2_body},
        'lhand': {'l1': l1_lhand, 'l2_rmse': l2_lhand},
        'rhand': {'l1': l1_rhand, 'l2_rmse': l2_rhand},
    }

    # --- Reconstruction error in denormalized feature space ---
    orig_denorm = orig_norm * (std + 1e-10) + mean
    recon_denorm = recon_norm * (std + 1e-10) + mean

    l1_denorm = torch.mean(torch.abs(orig_denorm - recon_denorm)).item()
    l2_denorm = torch.sqrt(torch.mean((orig_denorm - recon_denorm) ** 2)).item()

    l1_body_d = torch.mean(torch.abs(orig_denorm[..., body_dims] - recon_denorm[..., body_dims])).item()
    l2_body_d = torch.sqrt(torch.mean((orig_denorm[..., body_dims] - recon_denorm[..., body_dims]) ** 2)).item()

    l1_lhand_d = torch.mean(torch.abs(orig_denorm[..., 30:75] - recon_denorm[..., 30:75])).item()
    l2_lhand_d = torch.sqrt(torch.mean((orig_denorm[..., 30:75] - recon_denorm[..., 30:75]) ** 2)).item()

    l1_rhand_d = torch.mean(torch.abs(orig_denorm[..., 75:120] - recon_denorm[..., 75:120])).item()
    l2_rhand_d = torch.sqrt(torch.mean((orig_denorm[..., 75:120] - recon_denorm[..., 75:120]) ** 2)).item()

    results['reconstruction_error_denormalized'] = {
        'overall': {'l1': l1_denorm, 'l2_rmse': l2_denorm},
        'body': {'l1': l1_body_d, 'l2_rmse': l2_body_d},
        'lhand': {'l1': l1_lhand_d, 'l2_rmse': l2_lhand_d},
        'rhand': {'l1': l1_rhand_d, 'l2_rmse': l2_rhand_d},
    }

    # Store tensors for video rendering
    results['_orig_norm'] = orig_norm
    results['_recon_norm'] = recon_norm

    return results


def compute_aggregate_codebook_utilization(all_results):
    """Compute codebook utilization across all samples."""
    utilization = {}

    for part, nb_code in [('body', 96), ('lhand', 192), ('rhand', 192)]:
        all_codes = []
        for r in all_results:
            all_codes.extend(r['code_indices'][part])

        all_codes = np.array(all_codes)
        unique_codes = np.unique(all_codes)
        usage_counts = np.bincount(all_codes, minlength=nb_code)

        # Perplexity
        total = len(all_codes)
        prob = usage_counts / total
        prob_nonzero = prob[prob > 0]
        perplexity = float(np.exp(-np.sum(prob_nonzero * np.log(prob_nonzero))))

        dead_codes = np.where(usage_counts == 0)[0].tolist()

        utilization[part] = {
            'nb_code': nb_code,
            'unique_codes_used': len(unique_codes),
            'utilization_pct': float(len(unique_codes) / nb_code * 100),
            'total_tokens': total,
            'perplexity': perplexity,
            'num_dead_codes': len(dead_codes),
            'dead_codes': dead_codes,
            'usage_counts': usage_counts.tolist(),
        }

    return utilization


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_codebook_utilization(utilization, output_dir):
    """Plot codebook usage histograms for each VAE."""
    for part in ['body', 'lhand', 'rhand']:
        info = utilization[part]
        counts = np.array(info['usage_counts'])
        nb_code = info['nb_code']

        fig, ax = plt.subplots(figsize=(max(12, nb_code * 0.08), 5))

        colors = ['#d62728' if c == 0 else '#1f77b4' for c in counts]
        ax.bar(range(nb_code), counts, color=colors, width=0.8)

        ax.set_xlabel('Codebook Index')
        ax.set_ylabel('Usage Count')
        ax.set_title(
            f'{part.upper()} VAE Codebook Utilization\n'
            f'{info["unique_codes_used"]}/{nb_code} codes used '
            f'({info["utilization_pct"]:.1f}%), '
            f'Perplexity={info["perplexity"]:.1f}, '
            f'Dead codes={info["num_dead_codes"]} (red)'
        )
        ax.set_xlim(-0.5, nb_code - 0.5)

        plt.tight_layout()
        path = os.path.join(output_dir, f'codebook_utilization_{part}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")


def plot_quantization_error_distribution(all_results, output_dir):
    """Plot distribution of per-timestep L2 quantization errors."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, part in enumerate(['body', 'lhand', 'rhand']):
        all_means = [r['latent_l2_distance'][part]['mean'] for r in all_results]
        all_maxes = [r['latent_l2_distance'][part]['max'] for r in all_results]

        x = range(len(all_results))
        names = [r['name'][:20] for r in all_results]

        axes[idx].bar(x, all_means, color='#1f77b4', alpha=0.7, label='Mean L2')
        axes[idx].scatter(x, all_maxes, color='#d62728', marker='v', s=60, zorder=3, label='Max L2')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        axes[idx].set_ylabel('L2 Distance (512-dim latent)')
        axes[idx].set_title(f'{part.upper()} VAE - Latent Quantization Error')
        axes[idx].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, 'quantization_error_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_reconstruction_error(all_results, output_dir):
    """Plot per-sample, per-body-part reconstruction error."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    names = [r['name'][:20] for r in all_results]
    x = np.arange(len(all_results))
    width = 0.2

    for ax_idx, (space, title) in enumerate([
        ('reconstruction_error_normalized', 'Normalized Feature Space'),
        ('reconstruction_error_denormalized', 'Denormalized (Angle-Axis)')
    ]):
        body_vals = [r[space]['body']['l1'] for r in all_results]
        lhand_vals = [r[space]['lhand']['l1'] for r in all_results]
        rhand_vals = [r[space]['rhand']['l1'] for r in all_results]
        overall_vals = [r[space]['overall']['l1'] for r in all_results]

        axes[ax_idx].bar(x - 1.5 * width, body_vals, width, label='Body', color='#1f77b4')
        axes[ax_idx].bar(x - 0.5 * width, lhand_vals, width, label='L-Hand', color='#ff7f0e')
        axes[ax_idx].bar(x + 0.5 * width, rhand_vals, width, label='R-Hand', color='#2ca02c')
        axes[ax_idx].bar(x + 1.5 * width, overall_vals, width, label='Overall', color='#9467bd')

        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        axes[ax_idx].set_ylabel('L1 Error')
        axes[ax_idx].set_title(f'Reconstruction Error - {title}')
        axes[ax_idx].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, 'reconstruction_error_per_sample.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def render_comparison_video(name, orig_norm, recon_norm, mean, std, output_dir, fps, device):
    """Render side-by-side video: original (left) vs reconstructed (right)."""
    with torch.no_grad():
        orig_verts = feats2joints(orig_norm.to(device), mean.to(device), std.to(device))
        recon_verts = feats2joints(recon_norm.to(device), mean.to(device), std.to(device))

    orig_verts_np = orig_verts.cpu().numpy()
    recon_verts_np = recon_verts.cpu().numpy()

    save_path = os.path.join(output_dir, f'{name}_comparison.mp4')
    render_video_from_meshes(
        verts_list=recon_verts_np,
        faces=smpl_x.face,
        save_path=save_path,
        fps=fps,
        ref_verts_list=orig_verts_np,
    )
    print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VQVAE Reconstruction Analysis for Thai Sign Language")
    parser.add_argument("--ckpt_path", type=str, default="../pretrained/tokenizer.ckpt")
    parser.add_argument("--mean_path", type=str, default="../data/CSL-Daily/mean.pt")
    parser.add_argument("--std_path", type=str, default="../data/CSL-Daily/std.pt")
    parser.add_argument("--thai_data_dir", type=str, default="../data/Thai")
    parser.add_argument("--output_dir", type=str, default="./results/vqvae_analysis")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--render_videos", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Validate paths
    for path, desc in [(args.ckpt_path, "Checkpoint"), (args.mean_path, "Mean"), (args.std_path, "Std")]:
        if not os.path.exists(path):
            print(f"ERROR: {desc} file not found: {path}")
            sys.exit(1)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    if args.render_videos:
        videos_dir = os.path.join(args.output_dir, 'videos')
        os.makedirs(videos_dir, exist_ok=True)

    # Load models
    print("Loading VAE models...")
    body_vae, hand_vae, rhand_vae = load_vaes(args.ckpt_path, device)

    print("Loading normalization statistics...")
    mean, std = load_mean_std(args.mean_path, args.std_path, device)

    # Load Thai data
    print(f"\nLoading Thai samples from {args.thai_data_dir}...")
    samples = load_thai_poses(args.thai_data_dir)
    if not samples:
        print("ERROR: No Thai samples found!")
        sys.exit(1)

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    print(f"\nAnalyzing {len(samples)} sample(s)...\n")

    # Analyze each sample
    all_results = []
    for i, (name, raw_features) in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] Analyzing: {name}")

        result = analyze_single_sample(
            name, raw_features,
            body_vae, hand_vae, rhand_vae,
            mean, std, device
        )

        # Print key metrics
        print(f"  Commit loss  - body: {result['commit_loss']['body']:.4f}, "
              f"lhand: {result['commit_loss']['lhand']:.4f}, "
              f"rhand: {result['commit_loss']['rhand']:.4f}")
        print(f"  Perplexity   - body: {result['perplexity']['body']:.1f}/{BODY_VAE_CFG['code_num']}, "
              f"lhand: {result['perplexity']['lhand']:.1f}/{HAND_VAE_CFG['code_num']}, "
              f"rhand: {result['perplexity']['rhand']:.1f}/{HAND_VAE_CFG['code_num']}")
        print(f"  Latent L2    - body: {result['latent_l2_distance']['body']['mean']:.4f} (mean), "
              f"lhand: {result['latent_l2_distance']['lhand']['mean']:.4f}, "
              f"rhand: {result['latent_l2_distance']['rhand']['mean']:.4f}")
        print(f"  Recon L1     - body: {result['reconstruction_error_normalized']['body']['l1']:.4f}, "
              f"lhand: {result['reconstruction_error_normalized']['lhand']['l1']:.4f}, "
              f"rhand: {result['reconstruction_error_normalized']['rhand']['l1']:.4f}")

        # Render video
        if args.render_videos:
            print(f"  Rendering comparison video...")
            render_comparison_video(
                name, result['_orig_norm'], result['_recon_norm'],
                mean, std, videos_dir, args.fps, device
            )

        all_results.append(result)

    # Aggregate codebook utilization
    print("\nComputing aggregate codebook utilization...")
    utilization = compute_aggregate_codebook_utilization(all_results)
    for part in ['body', 'lhand', 'rhand']:
        info = utilization[part]
        print(f"  {part}: {info['unique_codes_used']}/{info['nb_code']} codes used "
              f"({info['utilization_pct']:.1f}%), perplexity={info['perplexity']:.1f}, "
              f"dead={info['num_dead_codes']}")

    # Generate plots
    print("\nGenerating plots...")
    plot_codebook_utilization(utilization, plots_dir)
    plot_quantization_error_distribution(all_results, plots_dir)
    plot_reconstruction_error(all_results, plots_dir)

    # Build JSON report (strip internal tensor fields)
    report = {
        'num_samples': len(all_results),
        'samples': [],
        'aggregate_codebook_utilization': utilization,
    }
    for r in all_results:
        sample_report = {k: v for k, v in r.items() if not k.startswith('_')}
        report['samples'].append(sample_report)

    report_path = os.path.join(args.output_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")
    print("Done!")


if __name__ == "__main__":
    main()
