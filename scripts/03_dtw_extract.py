#!/usr/bin/env python
"""
Extract DTW alignments (JPE and PA) + mesh thumbnails from a saved SOKE test run.

Replaces 01_SOKE/03_document/06_dtw_extract_example.py, which could not reproduce the hand
metrics: it passed wanted=None to l2_dist_align, so the cost averaged over all 144
joints and --part had no effect (lhand and rhand both returned 11.094398 on C0003 vs
recorded 13.526096 / 14.313930). mGPT/metrics/t2m.py:180-190 instead regresses 21
MANO-style hand joints from the *mesh vertices* via smpl_x.orig_hand_regressor.

For every sample this writes one .npz holding, for each of the 3 parts x 2 cost modes:
the local cost matrix C, the accumulated matrix D, the warping path, and the score.
Mesh thumbnails are rendered in the same pass, because the vertex arrays that both the
hand metric and the renderer need are ~20 MB per sequence and must not hit disk.

Run from inside SOKE/ (the --mean-path/--std-path defaults are relative to it):
    cd 01_SOKE/SOKE
    PYTHONPATH=. ../envs/bin/python scripts/03_dtw_extract.py \
        --results-dir results/mgpt/SOKE-Thai-Hand4WholePP-From-Scratch/test_rank_0 \
        --select best,worst --recorded-mode jpe \
        --out-dir dtw_viewer_data
"""
import argparse
import json
import os
import pickle
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

from mGPT.utils.human_models import smpl_x
from mGPT.metrics.dtw import dtw, l2_dist_align

# Fixed shape vector hard-coded in mGPT/data/H2S.py:116 -- every sample is the same body.
SHAPE = torch.tensor([[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
                       0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]])

PARTS = ('body', 'lhand', 'rhand')
MODES = (('jpe', 0), ('pa', None))   # (name, align_idx) -- see t2m.py:31-32


# --------------------------------------------------------------------------- feats

def load_mean_std(mean_path, std_path):
    """Replicates the slicing in mGPT/data/H2S.py:51-58 -> (133,) each."""
    def sl(v):
        v = v[(3 + 3 * 11):]                         # drop root + 11 lower-body joints
        return torch.cat([v[:-20], v[-10:]], dim=0)  # drop 20, keep last 10 (expr)
    return sl(torch.load(mean_path)), sl(torch.load(std_path))


class Coord:
    """feats -> (vertices, joints), replicating mGPT/data/H2S.py:109-127.

    Unlike human_models.get_coord this keeps one SMPL-X layer instead of
    copy.deepcopy(...).cuda() per call (human_models.py:230), and runs in chunks so a
    long sequence does not need the whole batch resident on a nearly-full GPU.
    """

    def __init__(self, mean, std, device, chunk=32):
        self.mean, self.std = mean, std
        self.device = torch.device(device)
        self.chunk = chunk
        self.layer = smpl_x.layer['neutral'].to(self.device)

    def __call__(self, feats):
        f = torch.as_tensor(feats).float()                       # (T,133)
        f = f * self.std + self.mean
        T = f.shape[0]
        f = torch.cat([torch.zeros(T, 36), f], dim=-1)           # 36 + 133 = 169
        shape = SHAPE.repeat(T, 1)
        verts, joints = [], []
        for a in range(0, T, self.chunk):
            b = min(a + self.chunk, T)
            c, sh = f[a:b].to(self.device), shape[a:b].to(self.device)
            z = torch.zeros((b - a, 3), device=self.device)      # eye poses
            with torch.no_grad():
                o = self.layer(betas=sh, body_pose=c[:, 3:66], global_orient=c[:, 0:3],
                               right_hand_pose=c[:, 111:156], left_hand_pose=c[:, 66:111],
                               jaw_pose=c[:, 156:159], leye_pose=z, reye_pose=z,
                               expression=c[:, 159:169])
            verts.append(o.vertices.cpu())
            joints.append(o.joints.cpu())
        return torch.cat(verts), torch.cat(joints)               # (T,10475,3) (T,144,3)


# --------------------------------------------------------------------------- metric

def run_dtw(part, mode, align_idx, joints_rst, joints_ref, verts_rst, verts_ref):
    """One DTW run, mirroring mGPT/metrics/t2m.py:171-196.

    All three parts take align_idx from the caller, matching t2m.py now that its
    left-hand hardcode is fixed. Both modes are always computed: the run's configured
    mode is the recorded metric, the other is the viewer's diagnostic.
    """
    if part == 'body':
        # rigid_align / the translation both run on all 144 joints; `wanted` subsets
        # only afterwards (l2_dist_align:92-94). Not a bug -- match it.
        wanted = np.array(smpl_x.joint_part2idx['upper_body'])   # 12 idx into 144
        dist = partial(l2_dist_align, wanted=wanted, align_idx=align_idx)
        x, y = joints_rst, joints_ref
    else:
        side = 'left' if part == 'lhand' else 'right'
        R = smpl_x.orig_hand_regressor[side]                     # (21, 10475)
        x = torch.matmul(R, verts_rst).float().numpy()
        y = torch.matmul(R, verts_ref).float().numpy()
        dist = partial(l2_dist_align, align_idx=align_idx)       # wanted=None
    score, C, D, path = dtw(x, y, dist)
    return float(score), C.astype(np.float32), D.astype(np.float32), \
        np.asarray(path, dtype=np.int32)


# --------------------------------------------------------------------------- render

UPPER_GROUPS = ['neck', 'head', 'spine1', 'spine2',
                'leftShoulder', 'rightShoulder', 'leftArm', 'rightArm',
                'leftForeArm', 'rightForeArm',
                'leftHand', 'leftHandIndex1', 'rightHand', 'rightHandIndex1']

# SOKE/scripts/vis_mesh.py:88-90 applies Rx(180) because it renders into an image with
# y-down pixel coordinates. We use an orthographic camera in world space, where the
# SMPL-X output is already y-up, so no flip is applied -- adding one puts the head at
# the bottom of the tile.
#
# The 22 deg yaw is not decoration. Dead-on front view flattens the hands into the torso
# silhouette at 128 px, so a moving hand and a frozen one look identical -- which would
# defeat the point of the viewer. A slight three-quarter turn separates them and makes
# individual fingers legible, while staying close enough to frontal to read the sign.
YAW_DEG = 22.0


def _yaw(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0.0, np.sin(a)],
                     [0.0, 1.0, 0.0],
                     [-np.sin(a), 0.0, np.cos(a)]], dtype=np.float32)


VIEW = _yaw(YAW_DEG)


def vertex_normals(v, faces):
    """Per-vertex normals; pyrender leaves a Primitive unshaded without them."""
    tri = v[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    return (vn / np.maximum(ln, 1e-8)).astype(np.float32)


def upper_body_vertex_idx():
    idx = []
    for g in UPPER_GROUPS:
        if g in smpl_x.vertex_idx:
            idx.extend(smpl_x.vertex_idx[g])
    return np.array(sorted(set(idx)), dtype=np.int64)


def fit_camera(vert_sets, crop_idx, pad=1.12):
    """One orthographic camera for every frame of both tracks of a sample.

    A per-frame camera would rescale the view as the body moves and make a frozen hold
    look like motion -- which is exactly the thing the viewer exists to detect.
    """
    lo = np.full(3, np.inf, np.float32)
    hi = np.full(3, -np.inf, np.float32)
    for v in vert_sets:
        s = v[:, crop_idx] @ VIEW.T
        lo = np.minimum(lo, s.reshape(-1, 3).min(0))
        hi = np.maximum(hi, s.reshape(-1, 3).max(0))
    centre = (lo + hi) / 2.0
    half = float(max(hi[0] - lo[0], hi[1] - lo[1])) / 2.0 * pad
    return centre, max(half, 1e-3), float(hi[2] - lo[2])


def look_rotation(direction):
    """Rotation whose -Z axis points from `direction` back to the origin."""
    z = direction / np.linalg.norm(direction)          # light sits along +direction
    up = np.array([0.0, 1.0, 0.0], np.float32)
    if abs(float(z @ up)) > 0.99:
        up = np.array([0.0, 0.0, 1.0], np.float32)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1).astype(np.float32)


class Renderer:
    def __init__(self, tile):
        import pyrender
        self.pyrender = pyrender
        self.faces = np.asarray(smpl_x.layer['neutral'].faces, dtype=np.int32)
        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=0.65, alphaMode='OPAQUE',
            baseColorFactor=(1.0, 0.97, 0.92, 1.0))
        self.r = pyrender.OffscreenRenderer(tile, tile, point_size=1.0)
        self.tile = tile

    def frames(self, verts, centre, half, depth):
        """verts (T,10475,3) torch/np -> iterator of (tile,tile,3) uint8."""
        pyrender = self.pyrender
        cam = pyrender.OrthographicCamera(xmag=half, ymag=half,
                                          znear=0.01, zfar=max(4.0 * depth, 10.0))
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [centre[0], centre[1], centre[2] + max(2.0 * depth, 5.0)]
        v_all = np.asarray(verts, dtype=np.float32) @ VIEW.T
        for v in v_all:
            prim = pyrender.Primitive(positions=v, normals=vertex_normals(v, self.faces),
                                      indices=self.faces, material=self.material, mode=4)
            scene = pyrender.Scene(ambient_light=(0.28, 0.28, 0.29),
                                   bg_color=(1.0, 1.0, 1.0, 1.0))
            scene.add(pyrender.Mesh(primitives=[prim]))
            scene.add(cam, pose=pose)
            # key light over the viewer's shoulder, fill from the opposite side, so
            # depth is readable at 128 px without blowing out the highlights
            for offset, power in (((0.5, 0.8, 1.0), 7.5), ((-0.9, 0.1, 0.5), 2.6)):
                lp = np.eye(4, dtype=np.float32)
                lp[:3, :3] = look_rotation(np.array(offset, np.float32))
                lp[:3, 3] = centre + np.array(offset, np.float32) * max(4.0 * depth, 4.0)
                scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=power),
                          pose=lp)
            colour, _ = self.r.render(scene)
            yield colour[:, :, :3]

    def close(self):
        self.r.delete()


def write_atlas(tiles, path, tile, quality, cols=16):
    from PIL import Image
    n = len(tiles)
    rows = (n + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tile, rows * tile), (255, 255, 255))
    for k, t in enumerate(tiles):
        sheet.paste(Image.fromarray(t), ((k % cols) * tile, (k // cols) * tile))
    sheet.save(path, quality=quality, optimize=True)
    return cols, rows


# --------------------------------------------------------------------------- resume

def is_complete(npz_path, want_render):
    """True only if this sample is fully done and safe to skip.

    Deliberately strict. A walltime kill or a full disk leaves a truncated .npz that
    np.load will happily open; treating that as finished would bake a half-written
    sample into the viewer with no error anywhere. Anything short of every expected
    key plus both atlas files on disk is recomputed.
    """
    if not npz_path.exists():
        return False
    try:
        with np.load(npz_path, allow_pickle=True) as d:
            for mode, _ in MODES:
                for part in PARTS:
                    for pre in ('C', 'D', 'path', 'score'):
                        if f'{pre}_{part}_{mode}' not in d:
                            return False
            if not want_render:
                return True
            for track in ('ref', 'gen'):
                if f'atlas_{track}' not in d:
                    return False
                jpg = npz_path.parent / f'{npz_path.stem}_{track}.jpg'
                if not jpg.exists() or jpg.stat().st_size == 0:
                    return False
    except Exception:
        return False        # unreadable or truncated -> redo it
    return True


# --------------------------------------------------------------------------- samples

def discover_ranks(results_dir, split='test'):
    """Return the rank dirs under `results_dir`, or itself if it *is* one.

    A DDP test run writes one `<split>_rank_N/` per GPU, each with its own .pkl files
    AND its own test_scores.json covering only that rank's share. Ranking samples
    against a single rank therefore picks that rank's extremes, not the run's -- so a
    run dir is expanded to all of its ranks and the scores merged.
    """
    rd = Path(results_dir)
    if (rd / 'test_scores.json').exists():
        return [rd]
    ranks = sorted(d for d in rd.glob(f'{split}_rank_*')
                   if (d / 'test_scores.json').exists())
    return ranks


def load_scores(results_dir, split='test'):
    """Merged {key: (rank_dir, rec)} across every rank, plus a printed breakdown.

    DistributedSampler pads the final batch by repeating samples so all ranks get
    equal work, so the same key legitimately appears in more than one rank. Keep the
    first and count the rest -- letting a later rank silently overwrite would make the
    reported score depend on glob order.
    """
    ranks = discover_ranks(results_dir, split)
    if not ranks:
        print(f'scores     : no {split}_scores.json under {results_dir}')
        return {}
    merged, dupes = {}, 0
    for d in ranks:
        rec = json.load(open(d / 'test_scores.json'))
        new = 0
        for k, v in rec.items():
            if k in merged:
                dupes += 1
            else:
                merged[k] = (d, v)
                new += 1
        print(f'  {d.name:16s} {len(rec):5d} samples ({new} new)')
    if len(ranks) > 1 or dupes:
        print(f'scores     : {len(merged)} unique keys across {len(ranks)} rank(s)'
              + (f', deduped {dupes} seen in >1 rank' if dupes else ''))
    return merged


def src_of(rec):
    """Dataset id lives only in the metric-name prefix; pkls carry no dataset field."""
    for k in rec:
        if '_DTW' in k:
            return k.split('_DTW')[0]
    return 'unknown'


def select_keys(scores, spec):
    """spec like 'best,median,worst' -> those picks per dataset, ranked by body score.

    Ranks over the merged multi-rank set, so 'worst' is the run's worst rather than
    some single GPU's worst.
    """
    want = [s.strip() for s in spec.split(',') if s.strip()]
    by_src = {}
    for key, (_, rec) in scores.items():
        s = src_of(rec)
        v = rec.get(f'{s}_DTW_MPJPE_PA_body')
        if v is not None:
            by_src.setdefault(s, []).append((v, key))
    out = []
    for s in sorted(by_src):
        items = sorted(by_src[s])
        picks = {'best': 0, 'median': len(items) // 2, 'worst': len(items) - 1}
        for w in want:
            if w in picks:
                out.append(items[picks[w]][1])
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True,
                    help='dir holding <key>.pkl and test_scores.json')
    ap.add_argument('--keys', default=None, help='comma-separated sample ids')
    ap.add_argument('--keys-file', default=None, help='file with one sample id per line')
    ap.add_argument('--select', default=None,
                    help="pick per dataset by body score, e.g. 'best,median,worst'")
    ap.add_argument('--recorded-mode', default='jpe', choices=['jpe', 'pa'],
                    help="the run's METRIC.DTW_ALIGN_MODE; decides which computed mode "
                         'is reconciled against test_scores.json')
    ap.add_argument('--split', default='test',
                    help='which {split}_rank_* dirs to merge when --results-dir is a run dir')
    ap.add_argument('--run-tag', default='',
                    help='prefix for output files and the run label in the viewer. '
                         'Required when several runs share one --out-dir: sample keys '
                         'are NOT unique across runs (the Thai clips appear in both the '
                         'Thai and the 4-dataset runs) and would overwrite each other.')
    ap.add_argument('--force', action='store_true',
                    help='re-extract samples that are already complete')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--chunk', type=int, default=32, help='SMPL-X frames per forward')
    ap.add_argument('--tile', type=int, default=128)
    ap.add_argument('--jpeg-quality', type=int, default=72)
    ap.add_argument('--no-render', action='store_true', help='scores only, no thumbnails')
    ap.add_argument('--allow-mismatch', action='store_true')
    ap.add_argument('--mean-path', default='../data/CSL-Daily/mean.pt')
    ap.add_argument('--std-path', default='../data/CSL-Daily/std.pt')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()

    rd = Path(a.results_dir)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores = load_scores(rd, a.split)

    if a.keys:
        keys = [k.strip() for k in a.keys.split(',') if k.strip()]
    elif a.keys_file:
        keys = [l.strip() for l in open(a.keys_file) if l.strip()]
    elif a.select:
        keys = select_keys(scores, a.select)
    else:
        ap.error('one of --keys / --keys-file / --select is required')
    tag = a.run_tag.strip()
    stem_of = (lambda k: f'{tag}__{k.split("/")[-1]}') if tag else \
              (lambda k: k.split('/')[-1])
    print(f'samples    : {len(keys)}' + (f'   run-tag: {tag}' if tag else ''))

    mean, std = load_mean_std(a.mean_path, a.std_path)
    coord = Coord(mean, std, a.device, a.chunk)
    crop_idx = upper_body_vertex_idx()
    renderer = None if a.no_render else Renderer(a.tile)

    failures, skipped, done = [], 0, 0
    for n, key in enumerate(keys, 1):
        stem = stem_of(key)
        # Check completeness BEFORE the SMPL-X forward -- checking after would run the
        # expensive part anyway and make the skip worthless.
        if not a.force and is_complete(out / f'{stem}.npz', renderer is not None):
            print(f'[{n}/{len(keys)}] {key}: SKIP (complete)')
            skipped += 1
            continue

        rank_dir, rec = scores.get(key, (rd, {}))
        # phoenix keys embed the split ('test/25April_...') but base.py:64 writes the
        # pkl as key.split('/')[-1].
        pkl = rank_dir / f'{key.split("/")[-1]}.pkl'
        if not pkl.exists():
            print(f'[{n}/{len(keys)}] {key}: MISSING {pkl}')
            failures.append(key)
            continue
        d = pickle.load(open(pkl, 'rb'))
        src = src_of(rec) if rec else 'unknown'

        t0 = time.time()
        v_rst, j_rst = coord(d['feats_rst'])
        v_ref, j_ref = coord(d['feats_ref'])
        jr = j_rst.numpy()
        jf = j_ref.numpy()
        N, M = jr.shape[0], jf.shape[0]
        print(f'[{n}/{len(keys)}] {key}  src={src}  N={N} M={M}  '
              f'smplx {time.time()-t0:.1f}s')

        payload = {'key': key, 'text': d['text'], 'src': src, 'N': N, 'M': M,
                   'recorded_mode': a.recorded_mode, 'run': tag or '-'}
        ok = True
        for mode, align_idx in MODES:
            for part in PARTS:
                t1 = time.time()
                score, C, D, path = run_dtw(part, mode, align_idx, jr, jf, v_rst, v_ref)
                payload[f'C_{part}_{mode}'] = C
                payload[f'D_{part}_{mode}'] = D
                payload[f'path_{part}_{mode}'] = path
                payload[f'score_{part}_{mode}'] = score
                line = (f'    {mode:3s} {part:5s} score {score:10.6f}  '
                        f'L={path.shape[1]:4d}  {time.time()-t1:4.1f}s')
                # t2m.py:182 used to hardcode align_idx=0 for lhand, which made the
                # recorded lhand number the jpe one whatever DTW_ALIGN_MODE said. That
                # is fixed, so all three parts now follow the config. A test_scores.json
                # written by a 'pa' run from BEFORE that fix would need the old rule
                # (`or (part == 'lhand' and mode == 'jpe')`) -- no such run exists, since
                # every recorded run to date is 'jpe', where the two rules agree anyway.
                is_recorded = (mode == a.recorded_mode)
                r = rec.get(f'{src}_DTW_MPJPE_PA_{part}')
                if is_recorded and r is not None:
                    match = abs(r - score) < 1e-3
                    payload[f'recorded_{part}'] = r
                    payload[f'match_{part}'] = match
                    line += f'  recorded {r:10.6f}  {"MATCH" if match else "MISMATCH"}'
                    ok &= match
                print(line)

        if not ok and not a.allow_mismatch:
            print(f'    -> refusing {key}: score does not reconcile with test_scores.json')
            failures.append(key)
            continue

        if renderer is not None:
            t2 = time.time()
            centre, half, depth = fit_camera([v_ref.numpy(), v_rst.numpy()], crop_idx)
            for track, verts in (('ref', v_ref), ('gen', v_rst)):
                tiles = list(renderer.frames(verts, centre, half, depth))
                cols, rows = write_atlas(tiles, out / f'{stem}_{track}.jpg',
                                         a.tile, a.jpeg_quality)
                payload[f'atlas_{track}'] = np.array([len(tiles), cols, rows, a.tile])
            print(f'    render {2*(N+M)//2} frames  {time.time()-t2:.1f}s')

        np.savez_compressed(out / f'{stem}.npz', **payload)
        done += 1

    if renderer is not None:
        renderer.close()
    print(f'\nwrote      : {out}')
    print(f'summary    : extracted {done}, skipped {skipped}, failed {len(failures)}')
    if failures:
        print(f'failed     : {len(failures)} -> {failures}')
        sys.exit(1)


if __name__ == '__main__':
    main()
