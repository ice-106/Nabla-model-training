"""
Convert folders of per-frame SMPL-X .pkl files into .mp4 videos.

Unlike convert_pkls_to_mp4.py (which takes single pkl files containing
pre-computed features), this script handles a directory tree where each
subfolder represents one sign-language clip and contains numbered per-frame
pkl files (e.g. 000001.pkl, 000002.pkl, ...).

Each pkl stores raw SMPL-X parameters:
  smplx_root_pose (3,), smplx_body_pose (63,), smplx_lhand_pose (45,),
  smplx_rhand_pose (45,), smplx_jaw_pose (3,), smplx_shape (10,),
  smplx_expr (10,)

Usage:
  conda activate pytorchmacos
  python scripts/convert_pkls_folder_to_mp4.py \
      --input_dir /Users/vikimark/Downloads/Thai/Poses \
      --output_dir /tmp/thai_videos \
      --fps 20

Example structure:
  input_dir/
    subfolder_A/
      000001.pkl
      000002.pkl
      ...
    subfolder_B/
      000001.pkl
      ...

Output:
  output_dir/
    subfolder_A.mp4
    subfolder_B.mp4
"""

import os
# Set headless rendering before any OpenGL import
os.environ["PYOPENGL_PLATFORM"] = "egl"

import pickle
import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

from mGPT.utils.human_models import smpl_x, get_coord
from mGPT.utils.render_utils import render_video_from_meshes


def load_frames_from_folder(folder_path):
    """Load all per-frame SMPL-X pkl files from a folder, sorted by filename.

    Returns:
        dict with keys: root_pose, body_pose, lhand_pose, rhand_pose,
                        jaw_pose, shape, expr  — each (T, D) numpy arrays.
        T = number of valid frames loaded.
    Returns None if the folder contains no valid pkl files.
    """
    pkl_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    )
    if not pkl_files:
        return None

    frames = {
        "root_pose": [],
        "body_pose": [],
        "lhand_pose": [],
        "rhand_pose": [],
        "jaw_pose": [],
        "shape": [],
        "expr": [],
    }

    key_map = {
        "smplx_root_pose": "root_pose",
        "smplx_body_pose": "body_pose",
        "smplx_lhand_pose": "lhand_pose",
        "smplx_rhand_pose": "rhand_pose",
        "smplx_jaw_pose": "jaw_pose",
        "smplx_shape": "shape",
        "smplx_expr": "expr",
    }

    for pkl_file in pkl_files:
        pkl_path = os.path.join(folder_path, pkl_file)
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            tqdm.write(f"  Warning: Could not load {pkl_path}: {e}")
            continue

        if not isinstance(data, dict):
            tqdm.write(f"  Warning: Unexpected data type in {pkl_path}: {type(data)}")
            continue

        # Check all required keys exist
        missing = [k for k in key_map if k not in data]
        if missing:
            tqdm.write(f"  Warning: Missing keys {missing} in {pkl_path}, skipping frame.")
            continue

        for src_key, dst_key in key_map.items():
            frames[dst_key].append(data[src_key])

    if not frames["root_pose"]:
        return None

    # Stack into (T, D) arrays
    for k in frames:
        frames[k] = np.stack(frames[k], axis=0)

    return frames


def frames_to_vertices(frames, device, batch_size=32):
    """Convert assembled SMPL-X parameters to mesh vertices.

    Args:
        frames: dict with numpy arrays, each (T, D)
        device: torch device
        batch_size: process this many frames at a time to avoid OOM

    Returns:
        vertices: numpy array (T, V, 3)
    """
    T = frames["root_pose"].shape[0]

    root_pose = torch.from_numpy(frames["root_pose"]).float().to(device)
    body_pose = torch.from_numpy(frames["body_pose"]).float().to(device)
    lhand_pose = torch.from_numpy(frames["lhand_pose"]).float().to(device)
    rhand_pose = torch.from_numpy(frames["rhand_pose"]).float().to(device)
    jaw_pose = torch.from_numpy(frames["jaw_pose"]).float().to(device)
    shape = torch.from_numpy(frames["shape"]).float().to(device)
    expr = torch.from_numpy(frames["expr"]).float().to(device)

    all_vertices = []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        verts, _ = get_coord(
            root_pose=root_pose[start:end],
            body_pose=body_pose[start:end],
            lhand_pose=lhand_pose[start:end],
            rhand_pose=rhand_pose[start:end],
            jaw_pose=jaw_pose[start:end],
            shape=shape[start:end],
            expr=expr[start:end],
        )
        all_vertices.append(verts.cpu().numpy())

    vertices = np.concatenate(all_vertices, axis=0)  # (T, V, 3)
    return vertices


def discover_sequence_folders(input_dir):
    """Recursively find all subfolders that contain .pkl files.

    Returns a sorted list of (folder_path, relative_name) tuples.
    relative_name is the path relative to input_dir, with '/' replaced by '_'
    for use as the output filename.
    """
    input_path = Path(input_dir).resolve()
    result = []

    for root, dirs, files in os.walk(input_dir):
        pkl_files = [f for f in files if f.endswith(".pkl")]
        if pkl_files:
            root_path = Path(root).resolve()
            rel = root_path.relative_to(input_path)
            # Use the relative path as the name (replace separators with _)
            name = str(rel).replace(os.sep, "_")
            result.append((str(root_path), name))

    result.sort(key=lambda x: x[1])
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert folders of per-frame SMPL-X PKL files to MP4 videos. "
            "Each subfolder containing .pkl files is treated as one clip."
        )
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Root directory containing subfolders of .pkl files."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save output .mp4 files."
    )
    parser.add_argument(
        "--fps", type=int, default=20,
        help="Frames per second for output video (default: 20)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Number of frames to process at once through SMPL-X (default: 32)."
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Max number of sequence folders to process. None = all."
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover all sequence folders
    print(f"Scanning for sequence folders in: {args.input_dir}")
    sequences = discover_sequence_folders(args.input_dir)
    print(f"Found {len(sequences)} sequence folder(s).")

    if not sequences:
        print("No sequence folders with .pkl files found. Exiting.")
        return

    if args.num_samples is not None:
        sequences = sequences[: args.num_samples]
        print(f"Limiting to first {args.num_samples} sequence(s).")

    faces = smpl_x.face

    for folder_path, seq_name in tqdm(sequences, desc="Sequences"):
        video_out_path = os.path.join(args.output_dir, f"{seq_name}.mp4")

        if os.path.exists(video_out_path):
            tqdm.write(f"Skipping {seq_name}: output already exists.")
            continue

        tqdm.write(f"Processing: {seq_name} ({folder_path})")

        try:
            # 1. Load per-frame pkl data
            frames = load_frames_from_folder(folder_path)
            if frames is None:
                tqdm.write(f"  No valid frames in {folder_path}. Skipping.")
                continue

            num_frames = frames["root_pose"].shape[0]
            tqdm.write(f"  Loaded {num_frames} frames.")

            # 2. Convert to mesh vertices
            vertices = frames_to_vertices(frames, args.device, args.batch_size)
            tqdm.write(f"  Vertices shape: {vertices.shape}")

            if vertices.shape[1] < 100:
                tqdm.write(f"  Error: vertex count too low ({vertices.shape[1]}). Skipping.")
                continue

            # 3. Render video
            render_video_from_meshes(
                verts_list=vertices,
                faces=faces,
                save_path=video_out_path,
                fps=args.fps,
            )
            tqdm.write(f"  Saved: {video_out_path}")

        except Exception as e:
            tqdm.write(f"  Failed to process {seq_name}: {e}")
            import traceback
            traceback.print_exc()

    print("Done.")


if __name__ == "__main__":
    main()
