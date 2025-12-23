import os
import argparse
import glob
# Set per-process environment variable for headless rendering before importing pyrender/trimesh
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import trimesh
from tqdm import tqdm
from mGPT.utils.render_utils import render_video_from_meshes
from mGPT.utils.human_models import smpl_x

def parse_args():
    parser = argparse.ArgumentParser(description="Batch render videos from folders containing .obj sequences")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory containing subfolders of .obj files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output .mp4 videos")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    return parser.parse_args()

def load_obj_sequence(folder_path):
    obj_files = sorted(glob.glob(os.path.join(folder_path, "*.obj")))
    if not obj_files:
        return None, None
    
    verts_list = []
    
    # Load first mesh to get faces
    first_mesh = trimesh.load(obj_files[0], process=False)
    faces = first_mesh.faces
    
    for f in obj_files:
        mesh = trimesh.load(f, process=False)
        verts_list.append(mesh.vertices)
            
    return np.array(verts_list), faces

def main():
    args = parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    # List all subdirectories in input_dir
    subdirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    subdirs.sort()
    
    print(f"Found {len(subdirs)} subdirectories to process in {args.input_dir}")
    
    for subdir in tqdm(subdirs):
        subdir_path = os.path.join(args.input_dir, subdir)
        print(f"Processing {subdir}...")
        
        try:
            verts, faces = load_obj_sequence(subdir_path)
            
            if verts is None:
                print(f"  Skipping {subdir}: No .obj files found.")
                continue
            
            # Use SMPL-X faces if not loaded from obj (safety fallback, though load_obj_sequence gets them)
            if faces is None or len(faces) == 0:
                 faces = smpl_x.face

            save_path = os.path.join(args.output_dir, f"{subdir}.mp4")
            
            render_video_from_meshes(
                verts_list=verts,
                faces=faces,
                save_path=save_path,
                fps=args.fps
            )
            
        except Exception as e:
            print(f"  Failed to process {subdir}: {e}")

if __name__ == "__main__":
    main()
