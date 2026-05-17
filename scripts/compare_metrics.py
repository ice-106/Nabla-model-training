"""
Compare Evaluation Metrics Between Two PKL Output Files
========================================================
Loads two .pkl files produced by the SOKE test pipeline and computes
Stage 1 (MRMetrics) or Stage 2 (TM2TMetrics) evaluation metrics
across all 6 pairwise combinations (4C2) of:
  A_ref, A_rst, B_ref, B_rst

The 6 pairs:
  1. A_ref vs A_rst   - File A reconstruction quality
  2. B_ref vs B_rst   - File B reconstruction quality
  3. A_rst vs B_ref   - File A prediction vs File B ground truth
  4. A_ref vs B_rst   - File A ground truth vs File B prediction
  5. A_ref vs B_ref   - Ground truth similarity
  6. A_rst vs B_rst   - Prediction similarity

Each pkl file contains:
  - feats_rst: (T, D) predicted SMPL-X motion features
  - feats_ref: (T, D) ground truth SMPL-X motion features
  - text: input text prompt

When mean/std and SMPL-X model are available, full joint/vertex-level
metrics (MPVPE, MPJPE, DTW-MPJPE) are computed. Otherwise, feature-level
metrics are used as a fallback.

Usage:
  python compare_pkl_metrics.py fileA.pkl fileB.pkl --stage 1
  python compare_pkl_metrics.py fileA.pkl fileB.pkl --stage 2 --mean_path ../data/How2Sign/mean.pt --std_path ../data/How2Sign/std.pt
  python compare_pkl_metrics.py  # interactive mode
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
from functools import partial


# ─── Terminal colors ────────────────────────────────────────────────────────

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def header(title):
    w = 70
    print(f"\n{BOLD}{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}{RESET}")


def section(title):
    print(f"\n{BOLD}{BLUE}--- {title} ---{RESET}")


def result_row(name, val_a, val_b, unit="mm"):
    """Print a comparison row with color-coded winner."""
    if isinstance(val_a, torch.Tensor):
        val_a = val_a.item()
    if isinstance(val_b, torch.Tensor):
        val_b = val_b.item()

    if abs(val_a - val_b) < 1e-6:
        ca, cb, winner = YELLOW, YELLOW, "  tie"
    elif val_a < val_b:
        ca, cb, winner = f"{GREEN}", f"{RED}", " <-- A"
    else:
        ca, cb, winner = f"{RED}", f"{GREEN}", "  B -->"

    print(f"  {name:<30s} {ca}{val_a:>10.4f}{RESET}  {cb}{val_b:>10.4f}{RESET}  {unit:<4s} {DIM}{winner}{RESET}")


# Labels and definitions for all 6 pairwise combinations
PAIR_DEFS = [
    ("A_ref vs A_rst", "File A recon quality",  "A_ref", "A_rst"),
    ("B_ref vs B_rst", "File B recon quality",  "B_ref", "B_rst"),
    ("A_rst vs B_ref", "A pred vs B GT",        "A_rst", "B_ref"),
    ("A_ref vs B_rst", "A GT vs B pred",        "A_ref", "B_rst"),
    ("A_ref vs B_ref", "GT similarity",         "A_ref", "B_ref"),
    ("A_rst vs B_rst", "Prediction similarity", "A_rst", "B_rst"),
]


def print_metric_table(title, pair_labels, pair_results, metric_keys, unit="mm",
                       higher_is_better=None):
    """
    Print a table with one row per pair, one column per metric.
    pair_labels:  list of str (length 6)
    pair_results: list of dict  (length 6)
    metric_keys:  list of str   (columns)
    higher_is_better: set of metric names where higher = better (e.g. cosine sim)
    """
    if higher_is_better is None:
        higher_is_better = set()

    section(title)

    # Determine column widths
    col_w = max(12, max(len(k) for k in metric_keys) + 2)
    label_w = 28

    # Header
    hdr = f"  {'Pair':<{label_w}s}"
    for k in metric_keys:
        hdr += f" {k:>{col_w}s}"
    hdr += f"  {'':>4s}"
    print(hdr)
    print(f"  {'─' * (label_w + (col_w + 1) * len(metric_keys) + 6)}")

    # Collect values for best-highlighting
    all_vals = {k: [] for k in metric_keys}
    for res in pair_results:
        for k in metric_keys:
            all_vals[k].append(res.get(k, float('nan')))

    for i, (label, res) in enumerate(zip(pair_labels, pair_results)):
        row = f"  {label:<{label_w}s}"
        for k in metric_keys:
            v = res.get(k, float('nan'))
            vals = [x for x in all_vals[k] if not np.isnan(x)]
            if len(vals) > 0 and not np.isnan(v):
                if k in higher_is_better:
                    best = max(vals)
                    worst = min(vals)
                else:
                    best = min(vals)
                    worst = max(vals)
                if abs(best - worst) < 1e-8:
                    c = YELLOW
                elif abs(v - best) < 1e-8:
                    c = GREEN
                elif abs(v - worst) < 1e-8:
                    c = RED
                else:
                    c = YELLOW
            else:
                c = DIM
            row += f" {c}{v:>{col_w}.4f}{RESET}"
        row += f"  {unit}"
        print(row)


def info_line(label, value):
    print(f"  {DIM}{label:<25s}{RESET} {value}")


# ─── PKL loading ────────────────────────────────────────────────────────────

def load_pkl(path):
    """Load a SOKE output pkl file."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in pkl, got {type(data)}")

    feats_rst = data.get('feats_rst')
    feats_ref = data.get('feats_ref')
    text = data.get('text', '')

    if feats_rst is None or feats_ref is None:
        raise ValueError(f"PKL missing feats_rst/feats_ref. Keys: {list(data.keys())}")

    if isinstance(feats_rst, np.ndarray):
        feats_rst = torch.from_numpy(feats_rst).float()
    if isinstance(feats_ref, np.ndarray):
        feats_ref = torch.from_numpy(feats_ref).float()

    # Ensure (T, D)
    if feats_rst.dim() == 1:
        feats_rst = feats_rst.unsqueeze(0)
    if feats_ref.dim() == 1:
        feats_ref = feats_ref.unsqueeze(0)

    return feats_rst, feats_ref, text


def print_pkl_info(label, feats_rst, feats_ref, text, path):
    """Print summary of a loaded pkl file."""
    section(f"File {label}: {os.path.basename(path)}")
    info_line("Path", path)
    info_line("Text", text if text else "(empty)")
    info_line("feats_rst shape", str(tuple(feats_rst.shape)))
    info_line("feats_ref shape", str(tuple(feats_ref.shape)))
    info_line("feats_rst range",
              f"[{feats_rst.min().item():.4f}, {feats_rst.max().item():.4f}]")
    info_line("feats_ref range",
              f"[{feats_ref.min().item():.4f}, {feats_ref.max().item():.4f}]")
    T_rst, T_ref = feats_rst.shape[0], feats_ref.shape[0]
    info_line("Length match" if T_rst == T_ref else "Length mismatch",
              f"rst={T_rst}, ref={T_ref}")


# ─── SMPL-X conversion ─────────────────────────────────────────────────────

def try_load_smplx_pipeline(mean_path, std_path, device):
    """
    Try to load the feats-to-joints conversion pipeline.
    Returns (feats2joints_fn, smpl_x) or (None, None) on failure.
    """
    try:
        from mGPT.utils.human_models import smpl_x, get_coord
        from mGPT.utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle
    except ImportError as e:
        print(f"  {YELLOW}Cannot import SMPL-X utilities: {e}{RESET}")
        return None, None

    if not os.path.exists(mean_path):
        print(f"  {YELLOW}Mean file not found: {mean_path}{RESET}")
        return None, None
    if not os.path.exists(std_path):
        print(f"  {YELLOW}Std file not found: {std_path}{RESET}")
        return None, None

    try:
        mean = torch.load(mean_path, map_location=device, weights_only=False)
        std = torch.load(std_path, map_location=device, weights_only=False)

        # Preprocessing as in scripts/convert_pkls_to_mp4.py / scripts/vis_mesh.py
        mean = mean[(3 + 3 * 11):]
        mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
        std = std[(3 + 3 * 11):]
        std = torch.cat([std[:-20], std[-10:]], dim=0)
    except Exception as e:
        print(f"  {YELLOW}Failed to load mean/std: {e}{RESET}")
        return None, None

    def feats2joints(features, rot6d=False):
        """Convert normalized features to (vertices, joints)."""
        if features.dim() == 2:
            features = features.unsqueeze(0)  # (1, T, D)
        features = features.to(device)
        B, T, D = features.shape

        m, s = mean, std
        if m.shape[0] > D:
            m = m[:D]
            s = s[:D]
        features = features * s + m

        if features.shape[-1] == 123:
            features = torch.cat([features, torch.zeros(B, T, 10, device=device)], dim=-1)

        if rot6d:
            expr = features[..., -10:]
            features_rot = features[..., :-10].view(B, T, -1, 6)
            features_rot = matrix_to_axis_angle(rotation_6d_to_matrix(features_rot))
            features_rot = features_rot.view(B, T, -1)
            features = torch.cat([features_rot, expr], dim=-1)

        zero_pose = torch.zeros(B, T, 36, device=device)
        shape_param = torch.tensor(
            [[[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
               0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]]],
            device=device
        ).repeat(B, T, 1).view(B * T, -1)

        features = torch.cat([zero_pose, features], dim=-1).view(B * T, -1)
        vertices, joints = get_coord(
            root_pose=features[..., 0:3], body_pose=features[..., 3:66],
            lhand_pose=features[..., 66:111], rhand_pose=features[..., 111:156],
            jaw_pose=features[..., 156:159], shape=shape_param,
            expr=features[..., 159:169]
        )
        # vertices: (B*T, V, 3), joints: (B*T, J, 3)
        vertices = vertices.view(B, T, -1, 3)
        joints = joints.view(B, T, -1, 3)
        return vertices, joints

    return feats2joints, smpl_x


# ─── Rigid alignment (standalone, no SMPL-X needed) ────────────────────────

def rigid_align_np(A, B):
    """Procrustes alignment of A onto B. Both (N, 3)."""
    A_mean = A.mean(axis=0)
    B_mean = B.mean(axis=0)
    Ac = A - A_mean
    Bc = B - B_mean
    H = Ac.T @ Bc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign = np.eye(3)
    sign[2, 2] = np.sign(d)
    R = Vt.T @ sign @ U.T
    var_A = np.sum(Ac ** 2)
    scale = np.trace(np.diag(S) @ sign) / var_A if var_A > 1e-10 else 1.0
    return scale * (Ac @ R.T) + B_mean


def rigid_align_torch_batch(P, Q):
    """Batched Procrustes of P onto Q. Both (B, N, 3)."""
    Pm = P.mean(dim=1, keepdim=True)
    Qm = Q.mean(dim=1, keepdim=True)
    Pc = P - Pm
    Qc = Q - Qm
    H = Pc.transpose(1, 2) @ Qc
    U, S, Vh = torch.linalg.svd(H)
    d = torch.det(Vh.transpose(1, 2) @ U.transpose(1, 2))
    sign = torch.ones(P.shape[0], 3, device=P.device)
    sign[:, 2] = torch.sign(d)
    R = Vh.transpose(1, 2) @ torch.diag_embed(sign) @ U.transpose(1, 2)
    var_P = (Pc ** 2).sum(dim=(1, 2))
    sc = (S * sign).sum(dim=1) / var_P.clamp(min=1e-10)
    return sc[:, None, None] * (Pc @ R.transpose(1, 2)) + Qm


# ─── Feature-level metrics ─────────────────────────────────────────────────

def feature_l2(rst, ref):
    """Per-frame L2 distance in feature space, averaged. Both (T, D)."""
    T = min(rst.shape[0], ref.shape[0])
    diff = rst[:T] - ref[:T]
    return torch.sqrt((diff ** 2).sum(dim=-1)).mean()


def feature_smooth_l1(rst, ref):
    """SmoothL1 loss (same as used in Stage 1 reconstruction loss)."""
    T = min(rst.shape[0], ref.shape[0])
    return torch.nn.functional.smooth_l1_loss(rst[:T], ref[:T])


def feature_cosine_sim(rst, ref):
    """Mean cosine similarity per frame."""
    T = min(rst.shape[0], ref.shape[0])
    sim = torch.nn.functional.cosine_similarity(rst[:T], ref[:T], dim=-1)
    return sim.mean()


# ─── DTW (standalone) ──────────────────────────────────────────────────────

def dtw_distance(x, y, dist_func):
    """Dynamic Time Warping. x: (T1, ...), y: (T2, ...)."""
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist_func(x[i], y[j])
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i + 1, j], D0[i, j + 1])
    return D1[-1, -1]


def l2_dist_align(x, y, wanted=None, align_idx=None):
    """L2 distance with optional alignment. x, y: (N, 3) numpy."""
    if align_idx is None:
        x = rigid_align_np(x, y)
    else:
        x = x - x[align_idx:align_idx + 1] + y[align_idx:align_idx + 1]
    if wanted is not None:
        x = x[wanted]
        y = y[wanted]
    return np.mean(np.sqrt(((x - y) ** 2).sum(axis=1)))


# ─── Stage 1 metrics (joint/vertex level) ──────────────────────────────────

def compute_stage1_metrics(vertices, joints, label):
    """
    Compute Stage 1 MRMetrics for a single sample.
    vertices: dict with 'rst' (1,T1,V,3) and 'ref' (1,T2,V,3)
    joints: dict with 'rst' (1,T1,J,3) and 'ref' (1,T2,J,3)
    Truncates to the shorter sequence length when T1 != T2.
    Returns dict of metric_name -> value (in mm).
    """
    v_rst = vertices['rst'][0].cpu()  # (T1, V, 3)
    v_ref = vertices['ref'][0].cpu()  # (T2, V, 3)
    j_rst = joints['rst'][0].cpu()    # (T1, J, 3)
    j_ref = joints['ref'][0].cpu()    # (T2, J, 3)

    # Truncate to common length
    T = min(v_rst.shape[0], v_ref.shape[0])
    v_rst = v_rst[:T]
    v_ref = v_ref[:T]
    j_rst = j_rst[:T]
    j_ref = j_ref[:T]

    results = {}
    factor = 1000.0  # m -> mm

    # MPVPE_PA_all
    aligned = rigid_align_torch_batch(v_rst, v_ref)
    results['MPVPE_PA_all'] = torch.mean(
        torch.sqrt(((aligned - v_ref) ** 2).sum(dim=-1))
    ).item() * factor

    # MPVPE_all (root-aligned)
    try:
        from mGPT.utils.human_models import smpl_x
        pelvis_idx = smpl_x.J_regressor_idx['pelvis']
        root_rst = j_rst[:, pelvis_idx:pelvis_idx + 1]
        root_ref = j_ref[:, pelvis_idx:pelvis_idx + 1]
    except Exception:
        root_rst = j_rst[:, 0:1]
        root_ref = j_ref[:, 0:1]
    v_aligned = v_rst - root_rst + root_ref
    results['MPVPE_all'] = torch.mean(
        torch.sqrt(((v_aligned - v_ref) ** 2).sum(dim=-1))
    ).item() * factor

    # MPJPE_PA_body (using j14_regressor or first 14 joints)
    try:
        from mGPT.utils.human_models import smpl_x as sx
        j14_rst = torch.matmul(sx.j14_regressor, v_rst)
        j14_ref = torch.matmul(sx.j14_regressor, v_ref)
        j14_aligned = rigid_align_torch_batch(j14_rst, j14_ref)
        results['MPJPE_PA_body'] = torch.mean(
            torch.sqrt(((j14_aligned - j14_ref) ** 2).sum(dim=-1))
        ).item() * factor

        j14_root = j14_rst - root_rst + root_ref
        results['MPJPE_body'] = torch.mean(
            torch.sqrt(((j14_root - j14_ref) ** 2).sum(dim=-1))
        ).item() * factor
    except Exception:
        # Fallback: use raw joints
        j_aligned = rigid_align_torch_batch(j_rst, j_ref)
        results['MPJPE_PA_body'] = torch.mean(
            torch.sqrt(((j_aligned - j_ref) ** 2).sum(dim=-1))
        ).item() * factor

        j_root = j_rst - root_rst + root_ref
        results['MPJPE_body'] = torch.mean(
            torch.sqrt(((j_root - j_ref) ** 2).sum(dim=-1))
        ).item() * factor

    # Hand metrics
    try:
        from mGPT.utils.human_models import smpl_x as sx
        for side, key in [('left', 'lhand'), ('right', 'rhand')]:
            hand_v_idx = sx.hand_vertex_idx[f'{side}_hand']
            hv_rst = v_rst[:, hand_v_idx]
            hv_ref = v_ref[:, hand_v_idx]
            hv_aligned = rigid_align_torch_batch(hv_rst, hv_ref)
            results[f'MPVPE_PA_{key}'] = torch.mean(
                torch.sqrt(((hv_aligned - hv_ref) ** 2).sum(dim=-1))
            ).item() * factor

        # Average left/right for combined hand metric
        results['MPVPE_PA_hand'] = (results['MPVPE_PA_lhand'] + results['MPVPE_PA_rhand']) / 2.0

        for side, key in [('left', 'lhand'), ('right', 'rhand')]:
            hj_rst = torch.matmul(sx.orig_hand_regressor[side], v_rst)
            hj_ref = torch.matmul(sx.orig_hand_regressor[side], v_ref)
            hj_aligned = rigid_align_torch_batch(hj_rst, hj_ref)
            results[f'MPJPE_PA_{key}'] = torch.mean(
                torch.sqrt(((hj_aligned - hj_ref) ** 2).sum(dim=-1))
            ).item() * factor

        results['MPJPE_PA_hand'] = (results['MPJPE_PA_lhand'] + results['MPJPE_PA_rhand']) / 2.0

        # Face
        face_idx = sx.face_vertex_idx
        fv_rst = v_rst[:, face_idx]
        fv_ref = v_ref[:, face_idx]
        fv_aligned = rigid_align_torch_batch(fv_rst, fv_ref)
        results['MPVPE_PA_face'] = torch.mean(
            torch.sqrt(((fv_aligned - fv_ref) ** 2).sum(dim=-1))
        ).item() * factor

    except Exception:
        pass

    return results


# ─── Stage 2 metrics (DTW on joints) ───────────────────────────────────────

def compute_stage2_metrics(vertices, joints, smpl_x_obj=None):
    """
    Compute Stage 2 TM2TMetrics (DTW-MPJPE) for a single sample.
    joints: dict with 'rst' (1,T1,J,3) and 'ref' (1,T2,J,3)
    vertices: dict with 'rst' (1,T1,V,3) and 'ref' (1,T2,V,3)
    Returns dict of metric_name -> value (in mm).
    """
    j_rst = joints['rst'][0].cpu().numpy()    # (T1, J, 3)
    j_ref = joints['ref'][0].cpu().numpy()    # (T2, J, 3)
    v_rst = vertices['rst'][0].cpu()
    v_ref = vertices['ref'][0].cpu()

    results = {}
    factor = 1000.0

    # Body (upper body joints)
    try:
        from mGPT.utils.human_models import smpl_x as sx
        body_idx = sx.joint_part2idx['upper_body']
    except Exception:
        body_idx = list(range(min(30, j_rst.shape[1])))

    dist_func = partial(l2_dist_align, wanted=body_idx, align_idx=0)
    results['DTW_MPJPE_PA_body'] = dtw_distance(j_rst, j_ref, dist_func) * factor

    # Hands via hand regressor (mesh -> hand joints)
    try:
        from mGPT.utils.human_models import smpl_x as sx
        for side, key in [('left', 'lhand'), ('right', 'rhand')]:
            hj_rst = torch.matmul(sx.orig_hand_regressor[side], v_rst).float().numpy()
            hj_ref = torch.matmul(sx.orig_hand_regressor[side], v_ref).float().numpy()
            dist_func = partial(l2_dist_align, align_idx=0)
            results[f'DTW_MPJPE_PA_{key}'] = dtw_distance(hj_rst, hj_ref, dist_func) * factor
    except Exception:
        # Fallback: split joint array
        n_j = j_rst.shape[1]
        mid = n_j // 2
        q = (n_j - mid) // 2
        for name, idx_slice in [('lhand', list(range(mid, mid + q))),
                                ('rhand', list(range(mid + q, n_j)))]:
            dist_func = partial(l2_dist_align, wanted=idx_slice, align_idx=0)
            results[f'DTW_MPJPE_PA_{name}'] = dtw_distance(j_rst, j_ref, dist_func) * factor

    return results


# ─── Feature-level comparison (no SMPL-X needed) ───────────────────────────

def compute_feature_metrics(rst, ref):
    """Compute feature-space metrics. rst, ref: (T, D) tensors."""
    results = {}
    T = min(rst.shape[0], ref.shape[0])
    results['Feature_L2'] = feature_l2(rst, ref).item()
    results['Feature_SmoothL1'] = feature_smooth_l1(rst, ref).item()
    results['Feature_CosineSim'] = feature_cosine_sim(rst, ref).item()

    # Per-part feature analysis (body / lhand / rhand)
    D = rst.shape[-1]
    if D >= 123:
        # axis-angle: body=joints0-29->dims 0:36 (12 joints * 3),
        # but in the stored features the split is:
        # 0:36 upper body (12*3), 36:81 left hand (15*3), 81:126 right hand (15*3), ...
        # Actually from the SMPL-X layout and the code in mr.py:
        # smplx_part2idx: body=0:30, lhand=30:75, rhand=75:120, face=120:133
        # But the features are axis-angle, so the dimension mapping is different.
        # In axis-angle: each joint has 3 params, so:
        #   body ~0:36 (first 12 rotations), lhand ~36:81 (15 rotations), rhand ~81:126 (15 rotations)
        # However, the exact split depends on the config. Use rough thirds.
        body_end = min(36, D)
        hand_end = min(81, D)
        rhand_end = min(126, D)

        results['Feature_L2_body'] = feature_l2(rst[:, :body_end], ref[:, :body_end]).item()
        results['Feature_L2_lhand'] = feature_l2(rst[:, body_end:hand_end], ref[:, body_end:hand_end]).item()
        if rhand_end <= D:
            results['Feature_L2_rhand'] = feature_l2(rst[:, hand_end:rhand_end], ref[:, hand_end:rhand_end]).item()

    return results


# ─── DTW on raw features (no SMPL-X needed) ────────────────────────────────

def compute_feature_dtw(rst, ref):
    """DTW distance on raw feature vectors. rst: (T1, D), ref: (T2, D)."""
    rst_np = rst.numpy()
    ref_np = ref.numpy()

    def feat_l2(x, y):
        return np.sqrt(((x - y) ** 2).sum())

    return dtw_distance(rst_np, ref_np, feat_l2)


# ─── Build all 6 pairs from loaded data ────────────────────────────────────

def build_feat_pairs(rst_a, ref_a, rst_b, ref_b):
    """
    Return a dict mapping item names to feature tensors, and the list of
    6 pairs as (label, description, tensor_x, tensor_y).
    """
    items = {
        "A_ref": ref_a,
        "A_rst": rst_a,
        "B_ref": ref_b,
        "B_rst": rst_b,
    }
    pairs = []
    for label, desc, k1, k2 in PAIR_DEFS:
        pairs.append((label, desc, items[k1], items[k2]))
    return pairs


def build_smplx_pairs(converted):
    """
    converted: dict mapping item name -> (vertices, joints) each (1,T,N,3).
    Returns list of 6 pairs as (label, desc, verts_dict, joints_dict).
    """
    pairs = []
    for label, desc, k1, k2 in PAIR_DEFS:
        v1, j1 = converted[k1]
        v2, j2 = converted[k2]
        verts = {'rst': v1, 'ref': v2}
        joints = {'rst': j1, 'ref': j2}
        pairs.append((label, desc, verts, joints))
    return pairs


# ─── Main comparison logic ─────────────────────────────────────────────────

def compare(args):
    header("PKL Metric Comparison  (all 6 pairwise combinations)")

    # Load files
    rst_a, ref_a, text_a = load_pkl(args.file_a)
    rst_b, ref_b, text_b = load_pkl(args.file_b)

    print_pkl_info("A", rst_a, ref_a, text_a, args.file_a)
    print_pkl_info("B", rst_b, ref_b, text_b, args.file_b)

    stage = args.stage
    device = torch.device(args.device)

    # Build the 6 feature-level pairs
    feat_pairs = build_feat_pairs(rst_a, ref_a, rst_b, ref_b)

    # ── Pair legend ──
    section("Pairs (4C2 = 6)")
    for i, (label, desc, _, _) in enumerate(feat_pairs, 1):
        print(f"  {BOLD}{i}. {label:<22s}{RESET} {DIM}{desc}{RESET}")

    # ── Feature-level metrics (always available) ──
    pair_labels = [f"{i+1}. {p[0]}" for i, p in enumerate(feat_pairs)]
    pair_feat_results = []
    for label, desc, tx, ty in feat_pairs:
        pair_feat_results.append(compute_feature_metrics(tx, ty))

    # Determine which metric keys exist across all pairs
    all_feat_keys = []
    for r in pair_feat_results:
        for k in r:
            if k not in all_feat_keys:
                all_feat_keys.append(k)

    cosine_keys = {k for k in all_feat_keys if 'Cosine' in k}
    print_metric_table(
        "Feature-Level Metrics",
        pair_labels, pair_feat_results, all_feat_keys, unit="feat",
        higher_is_better=cosine_keys,
    )

    # ── Feature-level DTW (always available for Stage 2 or length mismatch) ──
    any_mismatch = any(tx.shape[0] != ty.shape[0] for _, _, tx, ty in feat_pairs)
    if stage == 2 or any_mismatch:
        pair_dtw_results = []
        for label, desc, tx, ty in feat_pairs:
            pair_dtw_results.append({'Feature_DTW': compute_feature_dtw(tx, ty)})
        print_metric_table(
            "Feature-Level DTW (handles length mismatch)",
            pair_labels, pair_dtw_results, ['Feature_DTW'], unit="feat",
        )

    # ── Try to load SMPL-X pipeline ──
    section("Loading SMPL-X pipeline")
    feats2joints_fn, smpl_x_obj = None, None
    if args.mean_path and args.std_path:
        feats2joints_fn, smpl_x_obj = try_load_smplx_pipeline(
            args.mean_path, args.std_path, device)

    has_smplx = feats2joints_fn is not None

    if has_smplx:
        print(f"  {GREEN}SMPL-X pipeline loaded. Computing joint/vertex metrics.{RESET}")
    else:
        print(f"  {YELLOW}SMPL-X not available. Using feature-level metrics only.{RESET}")
        if not args.mean_path:
            print(f"  {DIM}Hint: pass --mean_path and --std_path for full metrics{RESET}")

    # ── Joint/Vertex-level metrics (needs SMPL-X) ──
    if has_smplx:
        section("Converting features to joints/vertices")
        rot6d = args.rot6d
        converted = {}
        item_map = {"A_ref": ref_a, "A_rst": rst_a, "B_ref": ref_b, "B_rst": rst_b}
        try:
            for name, feats in item_map.items():
                v, j = feats2joints_fn(feats, rot6d=rot6d)
                converted[name] = (v, j)
            print(f"  {GREEN}Conversion successful.{RESET}")
            for name in item_map:
                v, j = converted[name]
                info_line(f"{name} verts", str(tuple(v.shape)))
                info_line(f"{name} joints", str(tuple(j.shape)))
        except Exception as e:
            print(f"  {RED}Conversion failed: {e}{RESET}")
            has_smplx = False

    if has_smplx:
        smplx_pairs = build_smplx_pairs(converted)

        if stage == 1:
            pair_mr_results = []
            for label, desc, verts, joints in smplx_pairs:
                pair_mr_results.append(compute_stage1_metrics(verts, joints, label))

            mr_keys = []
            for r in pair_mr_results:
                for k in r:
                    if k not in mr_keys:
                        mr_keys.append(k)

            print_metric_table(
                "Stage 1: Motion Reconstruction Metrics (MRMetrics)",
                pair_labels, pair_mr_results, mr_keys, unit="mm",
            )

        if stage == 2:
            print(f"\n  {DIM}Computing DTW (may be slow for long sequences)...{RESET}")
            pair_tm_results = []
            for label, desc, verts, joints in smplx_pairs:
                pair_tm_results.append(compute_stage2_metrics(verts, joints))

            tm_keys = []
            for r in pair_tm_results:
                for k in r:
                    if k not in tm_keys:
                        tm_keys.append(k)

            print_metric_table(
                "Stage 2: Text-to-Motion Metrics (TM2TMetrics / DTW)",
                pair_labels, pair_tm_results, tm_keys, unit="mm",
            )

    # ── Summary ──
    section("Summary")
    print(f"  File A: {os.path.basename(args.file_a)}")
    print(f"  File B: {os.path.basename(args.file_b)}")
    print(f"  Stage:  {stage}")
    print(f"  SMPLX:  {'Yes' if has_smplx else 'No (feature-level only)'}")
    print(f"  Pairs:  6 (all 4C2 combinations of A_ref, A_rst, B_ref, B_rst)")
    print(f"  {DIM}Lower is better for error metrics (L2, MPVPE, MPJPE, DTW).")
    print(f"  Higher is better for cosine similarity.")
    print(f"  Green = best across pairs, Red = worst across pairs.{RESET}")


# ─── Interactive mode ───────────────────────────────────────────────────────

def interactive():
    header("Interactive PKL Comparison")

    file_a = input(f"\n  {BOLD}Path to File A (.pkl): {RESET}").strip().strip("'\"")
    if not os.path.exists(file_a):
        print(f"  {RED}File not found: {file_a}{RESET}")
        return

    file_b = input(f"  {BOLD}Path to File B (.pkl): {RESET}").strip().strip("'\"")
    if not os.path.exists(file_b):
        print(f"  {RED}File not found: {file_b}{RESET}")
        return

    print(f"\n  {BOLD}Select stage:{RESET}")
    print("    1) Stage 1 - VAE reconstruction (MPVPE, MPJPE)")
    print("    2) Stage 2 - Text-to-motion (DTW-MPJPE)")
    stage_input = input(f"  > ").strip()
    stage = int(stage_input) if stage_input in ('1', '2') else 1

    mean_path = input(f"  {BOLD}Path to mean.pt (Enter to skip): {RESET}").strip().strip("'\"")
    std_path = ""
    if mean_path:
        std_path = input(f"  {BOLD}Path to std.pt: {RESET}").strip().strip("'\"")

    rot6d_input = input(f"  {BOLD}6D rotation format? [y/N]: {RESET}").strip().lower()
    rot6d = rot6d_input in ('y', 'yes')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = argparse.Namespace(
        file_a=file_a, file_b=file_b, stage=stage,
        mean_path=mean_path or None, std_path=std_path or None,
        rot6d=rot6d, device=device
    )
    compare(args)


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation metrics between two SOKE output .pkl files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python compare_pkl_metrics.py

  # Feature-level only (no SMPL-X needed)
  python compare_pkl_metrics.py output_a/sample.pkl output_b/sample.pkl --stage 1

  # Full metrics with SMPL-X
  python compare_pkl_metrics.py a.pkl b.pkl --stage 2 \\
      --mean_path ../data/How2Sign/mean.pt \\
      --std_path ../data/How2Sign/std.pt

  # With 6D rotation format
  python compare_pkl_metrics.py a.pkl b.pkl --stage 1 --rot6d \\
      --mean_path ../data/CSL-Daily/mean.pt \\
      --std_path ../data/CSL-Daily/std.pt
        """
    )
    parser.add_argument("file_a", nargs='?', help="Path to first .pkl file")
    parser.add_argument("file_b", nargs='?', help="Path to second .pkl file")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1,
                        help="Stage 1 (VAE/reconstruction) or Stage 2 (LM/text-to-motion)")
    parser.add_argument("--mean_path", type=str, default=None,
                        help="Path to mean.pt for feature denormalization")
    parser.add_argument("--std_path", type=str, default=None,
                        help="Path to std.pt for feature denormalization")
    parser.add_argument("--rot6d", action="store_true", default=False,
                        help="Features use 6D rotation format")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.file_a is None or args.file_b is None:
        interactive()
    else:
        if not os.path.exists(args.file_a):
            print(f"{RED}File not found: {args.file_a}{RESET}")
            sys.exit(1)
        if not os.path.exists(args.file_b):
            print(f"{RED}File not found: {args.file_b}{RESET}")
            sys.exit(1)
        compare(args)


if __name__ == "__main__":
    main()
