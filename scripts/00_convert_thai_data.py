import os
import numpy as np
import pickle
import glob
from tqdm import tqdm

import argparse

REQUIRED_KEYS = {
    'global_orient': 'smplx_root_pose',
    'body_pose': 'smplx_body_pose',
    'left_hand_pose': 'smplx_lhand_pose',
    'right_hand_pose': 'smplx_rhand_pose',
    'jaw_pose': 'smplx_jaw_pose',
    'betas': 'smplx_shape',
    'expression': 'smplx_expr'
}

def convert_frame(npz_path, output_path):
    try:
        data = np.load(npz_path)
        out_dict = {}
        
        for src_key, tgt_key in REQUIRED_KEYS.items():
            if src_key in data:
                arr = data[src_key]
                # Flatten the array to 1D
                out_dict[tgt_key] = arr.reshape(-1)
            else:
                # Warning is acceptable, but mostly we expect keys to exist
                pass
        
        # Save as pickle
        with open(output_path, 'wb') as f:
            pickle.dump(out_dict, f)
            
    except Exception as e:
        print(f"Error converting {npz_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert Thai dataset .npz to SOKE .pkl format")
    parser.add_argument('--input', '-i', type=str, required=True, help="Input directory containing video folders with smplx subfolders")
    parser.add_argument('--output', '-o', type=str, required=True, help="Output directory for poses")
    parser.add_argument('--smplx-folder', type=str, default='smplx', help="Name of the smplx subfolder (default: 'smplx')")
    args = parser.parse_args()

    source_dir = args.input
    target_dir = args.output
    smplx_folder_name = args.smplx_folder

    print(f"Scanning {source_dir}...")
    print(f"Outputting to {target_dir}")
    
    # Walk through source_dir
    for root, dirs, files in os.walk(source_dir):
        if smplx_folder_name in dirs:
            # This 'root' is likely the video folder (e.g., trimmed_..._C0003)
            video_name = os.path.basename(root)
            smplx_dir = os.path.join(root, smplx_folder_name)
            
            # Create target directory
            # If the user provides valid output dir, we usually append 'poses' inside it or use it as the poses root.
            # The previous script used `../data/Thai/poses`.
            # Let's assume the user passes the ROOT output folder, e.g. `data/Thai/poses`.
            # So we just append video_name.
            target_pose_dir = os.path.join(target_dir, video_name)
            os.makedirs(target_pose_dir, exist_ok=True)
            
            # Process all .npz files in smplx_dir
            npz_files = sorted(glob.glob(os.path.join(smplx_dir, '*.npz')))
            
            if not npz_files:
                continue
                
            print(f"Processing {video_name} ({len(npz_files)} frames)...")
            
            # Re-index from 0 based on sorted filename order
            for i, npz_file in enumerate(npz_files):
                target_name = f"{i:06d}.pkl"
                output_path = os.path.join(target_pose_dir, target_name)
                convert_frame(npz_file, output_path)

    print(f"\nConversion complete.")

if __name__ == "__main__":
    main()
