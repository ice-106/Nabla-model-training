import os
import pickle
import torch
import numpy as np
import trimesh
import argparse
from tqdm import tqdm
from mGPT.utils.human_models import smpl_x, get_coord
from mGPT.utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle

def feats2joints(features, mean, std, rot6d=False):
    #smpl2joints and drop lowerbody
    # Check dimensions
    # If features dim is 123, it might match mean/std, but be missing expr (10) for full SMPL-X
    # If mean/std are also 123, we perform transform, then we have 123-dim tensor.
    # Then we add zero_pose (36). Total 159.
    # get_coord needs expr at 159:169.
    # So we MUST pad features to 133 (if they are 123) AFTER normalization or BEFORE?
    # Usually normalization is applied to input features.
    
    # Let's handle dimensionality
    B, T, D = features.shape
    
    # Auto-adjust mean/std if they are larger than features (e.g. 133 vs 123)
    if mean.shape[0] > D:
        mean = mean[:D]
        std = std[:D]
    
    features = features * std + mean
    
    # Now features has size D (e.g. 123).
    # If D == 123, we need to pad 10 dims for expression?
    if features.shape[-1] == 123:
        # Pad with zeros for expression
        features = torch.cat([features, torch.zeros(B, T, 10).to(features)], dim=-1)
    
    zero_pose = torch.zeros(*features.shape[:-1], 36).to(features)
    shape_param = torch.tensor([[[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172, 
                            0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]]]).to(features)
    B, T = features.shape[:2]
    shape_param = shape_param.repeat(B, T, 1).view(B*T, -1)

    if rot6d:
            # 6d rotation to axis angle
            expr = features[..., -10:] #B,T,10
            features = features[..., :-10].view(B, T, -1, 6)
            features = matrix_to_axis_angle(rotation_6d_to_matrix(features))  #B,T,N,3
            features = features.view(B, T, -1)
            features = torch.cat([features, expr], dim=-1)

    features = torch.cat([zero_pose, features], dim=-1).view(B*T, -1)  #133+36=169
    vertices, joints = get_coord(root_pose=features[..., 0:3], body_pose=features[..., 3:66], 
                                    lhand_pose=features[..., 66:111], rhand_pose=features[..., 111:156], 
                                    jaw_pose=features[..., 156:159], shape=shape_param, 
                                    expr=features[..., 159:169])
    return vertices, joints

def main():
    parser = argparse.ArgumentParser(description="Convert SMPL-X PKL sequences to OBJ per frame.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .pkl files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output .obj files.")
    parser.add_argument("--mean_path", type=str, default="../data/CSL-Daily/mean.pt", help="Path to mean.pt")
    parser.add_argument("--std_path", type=str, default="../data/CSL-Daily/std.pt", help="Path to std.pt")
    parser.add_argument("--type", type=str, default="result", help="Render model prediction (result) or ground truth (reference)")
    # parser.add_argument("--rot6d", action="store_true", help="Whether using 6D rotation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use.")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    device = args.device

    if not os.path.exists(args.mean_path) or not os.path.exists(args.std_path):
        print(f"Error: Mean or Std file not found at {args.mean_path} or {args.std_path}")
        return

    # Load Mean and Std
    print(f"Loading mean/std from {args.mean_path} and {args.std_path}...")
    h2s_csl_mean = torch.load(args.mean_path, map_location=device)
    h2s_csl_std = torch.load(args.std_path, map_location=device)
    
    # Process mean/std as in vis_mesh.py
    h2s_csl_mean = h2s_csl_mean[(3+3*11):]
    h2s_csl_mean = torch.cat([h2s_csl_mean[:-20], h2s_csl_mean[-10:]], dim=0)
    h2s_csl_std = h2s_csl_std[(3+3*11):]
    h2s_csl_std = torch.cat([h2s_csl_std[:-20], h2s_csl_std[-10:]], dim=0)

    import csv

    os.makedirs(output_dir, exist_ok=True)
    
    pkl_files = [f for f in os.listdir(input_dir) if f.endswith(".pkl")]
    print(f"Found {len(pkl_files)} .pkl files.")

    metadata_list = []

    for pkl_file in tqdm(pkl_files):
        pkl_path = os.path.join(input_dir, pkl_file)
        file_basename = os.path.splitext(pkl_file)[0]
        
        # Determine specific output directory for this file
        file_out_dir = os.path.join(output_dir, file_basename)
        os.makedirs(file_out_dir, exist_ok=True)
        
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle dictionary structure
            features = None
            text_content = ""

            if isinstance(data, dict):
                # Check for key
                if 'feats_rst' in data and args.type == "result":
                    features = data['feats_rst']
                elif 'feats_ref' in data and args.type == "reference":
                    features = data['feats_ref']
                
                # Extract text
                if 'text' in data:
                    text_content = data['text']
            else:
                features = data
            
            if features is None:
                print(f"Skipping {pkl_file}: No suitable features found.")
                continue

            # Store metadata
            metadata_list.append([file_basename, text_content])

            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features)
            
            features = features.to(device).float()
            
            # Ensure batch dimension if missing (T, D) -> (1, T, D)
            if features.dim() == 2:
                features = features.unsqueeze(0)

            # Debug input shape
            if pkl_files.index(pkl_file) == 0:
                 print(f"Debug: Feature shape for first file: {features.shape}")
                 print(f"Debug: Mean shape: {h2s_csl_mean.shape}, Std shape: {h2s_csl_std.shape}")

            # Check if features match mean/std
            if features.shape[-1] != h2s_csl_mean.shape[-1]:
                print(f"Warning: Feature dim {features.shape[-1]} != Mean dim {h2s_csl_mean.shape[-1]}. Function feats2joints will attempt to handle it.")

            vertices, _ = feats2joints(features, h2s_csl_mean, h2s_csl_std, rot6d=False)
            if vertices.shape[0] == 0:
                 print(f"Error: No vertices returned for {pkl_file}")
                 continue
            
            vertices = vertices.cpu().numpy() # Keep all frames
            
            # Save Frames
            for i, vert in enumerate(vertices):
                if vert.shape[0] < 100:
                     print(f"Error: Vertices count too low ({vert.shape[0]}) for frame {i} in {pkl_file}. Skipping.")
                     continue
                
                mesh = trimesh.Trimesh(vertices=vert, faces=smpl_x.face, process=False)
                mesh.export(os.path.join(file_out_dir, f"{i:05d}.obj"))
                
        except Exception as e:
            print(f"Failed to process {pkl_file}: {e}")

    # Write CSV
    csv_path = os.path.join(output_dir, "metadata.csv")
    print(f"Writing metadata to {csv_path}...")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Text"])
        writer.writerows(metadata_list)

if __name__ == "__main__":
    main()
