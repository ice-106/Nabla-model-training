import os
import gzip
import pickle
import pandas as pd
import argparse
from tqdm import tqdm

bad_how2sign_ids = [
    '0DU7wWLK-QU_0-8-rgb_front', '0ICZi26jdaQ_28-5-rgb_front', '0vNfEYst_tQ_11-8-rgb_front', 
    '13X0vEMNm7M_8-5-rgb_front', '14weIYQswlE_23-8-rgb_front', '1B56XMJ-j1Q_13-8-rgb_front', 
    '1P0oKY4FNyI_0-8-rgb_front', '1dpRaxOTfZs_0-8-rgb_front', '1ei1kVTw23A_29-8-rgb_front', 
    '1spCnuBmWYk_0-8-rgb_front', '2-vXO7MMLJc_0-5-rgb_front', '21PbS6wnHtY_0-5-rgb_front', 
    '3tyfxL2wO-M_0-8-rgb_front', 'BpYDl3AO4B8_0-1-rgb_front', 'CH7AviIr0-0_14-8-rgb_front', 
    'CJ8RyW9pzKU_6-8-rgb_front', 'D0T7ho08Q3o_25-2-rgb_front', 'Db5SUQvNsHc_18-1-rgb_front', 
    'Eh697LCFjTw_0-3-rgb_front', 'F-p1IdedNbg_23-8-rgb_front', 'aUBQCNegrYc_13-1-rgb_front', 
    'cvn7htBA8Xc_9-8-rgb_front', 'czBrBQgZIuc_19-5-rgb_front', 'dbSAB8F8GYc_11-9-rgb_front', 
    'doMosV-zfCI_7-2-rgb_front', 'dvBdWGLzayI_10-8-rgb_front', 'eBrlZcccILg_26-3-rgb_front', 
    '39FN42e41r0_17-1-rgb_front', 'a4Nxq0QV_WA_9-3-rgb_front', 'fzrJBu2qsM8_11-8-rgb_front', 
    'g3Cc_1-V31U_12-3-rgb_front'
]

def main():
    parser = argparse.ArgumentParser(description="Extract Dataset Texts to CSV")
    parser.add_argument('--config', type=str, default='configs/soke.yaml', help='Path to soke.yaml config')
    parser.add_argument('--out_csv', type=str, default='dataset_texts.csv', help='Output CSV path')
    args = parser.parse_args()

    # Try loading roots from config, with fallback paths
    csl_root = '../data/CSL-Daily'
    phoenix_root = '../data/Phoenix_2014T'
    thai_root = '../data/Thai'
    h2s_root = '../data/How2Sign'

    if os.path.exists(args.config):
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(args.config)
            h2s_cfg = cfg.get('DATASET', {}).get('H2S', {})
            csl_root = h2s_cfg.get('CSL_ROOT', csl_root)
            phoenix_root = h2s_cfg.get('PHOENIX_ROOT', phoenix_root)
            thai_root = h2s_cfg.get('THAI_ROOT', thai_root)
            h2s_root = h2s_cfg.get('ROOT', h2s_root)
            print(f"Loaded config from {args.config}")
        except Exception as e:
            print(f"Failed to parse config with exception: {e}. Using default fallback paths.")
            
    # Also allow environment variables to override
    csl_root = os.environ.get('CSL_ROOT', csl_root)
    phoenix_root = os.environ.get('PHOENIX_ROOT', phoenix_root)
    thai_root = os.environ.get('THAI_ROOT', thai_root)
    h2s_root = os.environ.get('H2S_ROOT', h2s_root)

    print(f"Using roots:")
    print(f"  CSL:     {csl_root}")
    print(f"  Phoenix: {phoenix_root}")
    print(f"  Thai:    {thai_root}")
    print(f"  H2S:     {h2s_root}")

    all_data = []
    splits = ['train', 'val', 'test']

    # 1. Thai
    if thai_root and os.path.exists(thai_root):
        print(f"Processing Thai Dataset at {thai_root}")
        for split in splits:
            ann_file = 'val_vid.train' if split == 'train' else f'val_vid.{split}'
            ann_path = os.path.join(thai_root, ann_file)
            if os.path.exists(ann_path):
                with gzip.open(ann_path, 'rb') as f:
                    ann = pickle.load(f)
                for item in tqdm(ann, desc=f"Thai {split}"):
                    name = item['name']
                    text = item['text']
                    pose_dir = os.path.join(thai_root, 'poses', name)
                    num_frames = len(os.listdir(pose_dir)) if os.path.exists(pose_dir) else 0
                    all_data.append({
                        'filename': name,
                        'text': text,
                        'number_of_frame': num_frames,
                        'split': split,
                        'dataset': 'thai'
                    })

    # 2. CSL
    if csl_root and os.path.exists(csl_root):
        print(f"Processing CSL Dataset at {csl_root}")
        for split in splits:
            ann_file = 'csl_clean.train' if split == 'train' else f'csl_clean.{split}'
            ann_path = os.path.join(csl_root, ann_file)
            if os.path.exists(ann_path):
                with gzip.open(ann_path, 'rb') as f:
                    ann = pickle.load(f)
                for item in tqdm(ann, desc=f"CSL {split}"):
                    name = item['name']
                    text = item['text']
                    pose_dir = os.path.join(csl_root, 'poses', name)
                    num_frames = len(os.listdir(pose_dir)) if os.path.exists(pose_dir) else 0
                    all_data.append({
                        'filename': name,
                        'text': text,
                        'number_of_frame': num_frames,
                        'split': split,
                        'dataset': 'csl'
                    })

    # 3. Phoenix
    if phoenix_root and os.path.exists(phoenix_root):
        print(f"Processing Phoenix Dataset at {phoenix_root}")
        for split in splits:
            ann_file = 'phoenix14t.dev' if split == 'val' else f'phoenix14t.{split}'
            ann_path = os.path.join(phoenix_root, ann_file)
            if os.path.exists(ann_path):
                with gzip.open(ann_path, 'rb') as f:
                    ann = pickle.load(f)
                for item in tqdm(ann, desc=f"Phoenix {split}"):
                    name = item['name']
                    text = item['text']
                    pose_dir = os.path.join(phoenix_root, name)
                    num_frames = len(os.listdir(pose_dir)) if os.path.exists(pose_dir) else 0
                    all_data.append({
                        'filename': name,
                        'text': text,
                        'number_of_frame': num_frames,
                        'split': split,
                        'dataset': 'phoenix'
                    })
                    
    # 4. How2Sign
    if h2s_root and os.path.exists(h2s_root):
        print(f"Processing How2Sign Dataset at {h2s_root}")
        for split in splits:
            csv_path = os.path.join(h2s_root, split, 're_aligned', f'how2sign_realigned_{split}_preprocessed_fps.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['DURATION'] = df['END_REALIGNED'] - df['START_REALIGNED']
                df = df[df['DURATION'] < 30].reset_index(drop=True)
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"How2Sign {split}"):
                    name = row['SENTENCE_NAME']
                    if name in bad_how2sign_ids:
                        continue
                    text = row['SENTENCE']
                    pose_dir = os.path.join(h2s_root, split, 'poses', name)
                    num_frames = len(os.listdir(pose_dir)) if os.path.exists(pose_dir) else 0
                    # Apply fps sub-sampling logic for H2S if fps > 24
                    fps = row.get('fps', 0)
                    if fps > 24 and num_frames > 0:
                        import math
                        num_frames = int(math.floor(24 * num_frames / fps))
                    
                    all_data.append({
                        'filename': name,
                        'text': text,
                        'number_of_frame': num_frames,
                        'split': split,
                        'dataset': 'how2sign'
                    })

    if len(all_data) == 0:
        print("Warning: No datasets were found in the specified paths. Please ensure the paths are correct.")
        
    df_out = pd.DataFrame(all_data)
    if not df_out.empty:
        cols = ['filename', 'text', 'number_of_frame', 'split', 'dataset']
        df_out = df_out[cols]
    
    df_out.to_csv(args.out_csv, index=False)
    print(f"\nExtracted {len(df_out)} rows. Saved to {args.out_csv}.")

if __name__ == '__main__':
    main()
