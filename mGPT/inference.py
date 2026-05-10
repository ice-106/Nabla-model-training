import os
import torch
import torch.nn.functional as F
import pickle
from mGPT.utils.human_models import smpl_x
from mGPT.utils.render_utils import render_video_from_meshes

def run_inference(cfg, model, datamodule, logger):
    """
    Run inference on input texts to generate sign language motion.
    
    This function:
    1. Takes input text and generates motion tokens via model.lm.generate_conditional
    2. Decodes body/hand/rhand tokens via VAE decoders
    3. Assembles 133-dim features
    4. Converts to SMPL-X vertices via feats2joints
    5. Saves .pkl and renders video
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get inference configuration
    texts = cfg.INFER_TEXT
    src = cfg.INFER_SRC
    output_dir = cfg.INFER_OUTPUT_DIR
    fps = cfg.INFER_FPS
    
    if not texts:
        logger.error("No input texts provided. Use --text to specify input.")
        return
    
    if not src:
        logger.error("No source language specified. Use --src to specify (how2sign, csl, phoenix, thai).")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Inference mode: generating motion for {len(texts)} text(s)")
    logger.info(f"Source: {src}, Output dir: {output_dir}, FPS: {fps}")
    
    # Get the feats2joints function from datamodule
    feats2joints = model.feats2joints
    
    for i, text in enumerate(texts):
        logger.info(f"Processing text {i+1}/{len(texts)}: '{text}'")
        
        try:
            # Prepare input (batch of 1)
            batch_texts = [text]
            batch_src = [src]
            # If cfg.INFER_NAME is provided (e.g. from the demo notebook), use the
            # real video_id so KWS lookup in get_kw_strings hits name2kws.
            infer_names = cfg.get("INFER_NAME") if hasattr(cfg, "get") else None
            if infer_names and i < len(infer_names) and infer_names[i]:
                batch_name = [infer_names[i]]
            else:
                batch_name = [f"infer_{i}"]
            
            # Generate motion tokens using generate_conditional
            # Note: name parameter is used for keyword retrieval, unknown names return empty strings
            with torch.no_grad():
                gen_results = model.lm.generate_conditional(
                    batch_texts,
                    lengths=None,  # Will be inferred
                    stage='test',
                    tasks=None,
                    src=batch_src,
                    name=batch_name
                )
            
            outputs_tokens = gen_results['outputs_tokens']
            outputs_tokens_hand = gen_results.get('outputs_tokens_hand')
            outputs_tokens_rhand = gen_results.get('outputs_tokens_rhand')
            
            # Determine max length for padding
            max_len = max(map(len, outputs_tokens))
            if outputs_tokens_hand is not None:
                max_hand_len = max(map(len, outputs_tokens_hand))
                max_len = max(max_hand_len, max_len)
            if outputs_tokens_rhand is not None:
                max_rhand_len = max(map(len, outputs_tokens_rhand))
                max_len = max(max_rhand_len, max_len)
            max_len *= 4  # upsample factor
            
            # Get feature dimensions from VAE
            C = 133  # Standard feature dimension: body(30) + lhand(45) + rhand(45) + face(13)
            
            # Initialize output features
            feats_rst = torch.zeros(1, max_len, C).to(device)
            rst_len = max_len
            
            # Decode body tokens
            outputs_tokens[0] = torch.clamp(
                outputs_tokens[0], 0, model.vae.code_num - 1
            )
            
            if len(outputs_tokens[0]) > 1:
                motion = model.vae.decode(outputs_tokens[0])
                rst_len = motion.shape[1]
                motion = F.pad(motion, (0, 0, 0, max_len - motion.shape[1]), mode='replicate')
            else:
                motion = torch.zeros((1, max_len, model.vae.nfeats)).to(device)
                rst_len = 1
            
            # Fill body and face features
            feats_rst[0:1, :, :30] = motion[..., :30]
            feats_rst[0:1, :, -13:] = motion[..., 30:43]  # face features
            
            # Decode left hand tokens
            if outputs_tokens_hand is not None and hasattr(model, 'hand_vae'):
                outputs_tokens_hand[0] = torch.clamp(
                    outputs_tokens_hand[0], 0, model.hand_vae.code_num - 1
                )
                if len(outputs_tokens_hand[0]) > 1:
                    motion_hand = model.hand_vae.decode(outputs_tokens_hand[0])
                    rst_len = max(rst_len, motion_hand.shape[1])
                    motion_hand = F.pad(motion_hand, (0, 0, 0, max_len - motion_hand.shape[1]), mode='replicate')
                else:
                    motion_hand = torch.zeros((1, max_len, model.hand_vae.nfeats)).to(device)
                feats_rst[0:1, :, 30:30 + model.hand_vae.nfeats] = motion_hand
            
            # Decode right hand tokens
            if outputs_tokens_rhand is not None and hasattr(model, 'rhand_vae'):
                outputs_tokens_rhand[0] = torch.clamp(
                    outputs_tokens_rhand[0], 0, model.rhand_vae.code_num - 1
                )
                if len(outputs_tokens_rhand[0]) > 1:
                    motion_rhand = model.rhand_vae.decode(outputs_tokens_rhand[0])
                    rst_len = max(rst_len, motion_rhand.shape[1])
                    motion_rhand = F.pad(motion_rhand, (0, 0, 0, max_len - motion_rhand.shape[1]), mode='replicate')
                else:
                    motion_rhand = torch.zeros((1, max_len, model.rhand_vae.nfeats)).to(device)
                feats_rst[0:1, :, 75:75 + model.rhand_vae.nfeats] = motion_rhand
            
            # Trim to actual length
            feats_rst = feats_rst[:, :rst_len, :]
            
            # Convert features to SMPL-X vertices
            ret = feats2joints(feats_rst)  # Returns (vertices, joints) or vertices
            if isinstance(ret, tuple):
                vertices = ret[0]
            else:
                vertices = ret
            
            # Save .pkl file
            safe_text = "".join(c if c.isalnum() or c in ' -_' else '_' for c in text)[:50]
            pkl_filename = f"infer_{i}_{safe_text}.pkl"
            pkl_path = os.path.join(output_dir, pkl_filename)
            
            with open(pkl_path, 'wb') as f:
                pickle.dump({
                    'text': text,
                    'src': src,
                    'feats_rst': feats_rst.cpu().numpy(),
                    'length': rst_len
                }, f)
            logger.info(f"Saved features to {pkl_path}")
            
            # Render video
            video_filename = f"infer_{i}_{safe_text}.mp4"
            video_path = os.path.join(output_dir, video_filename)
            
            # Convert vertices to numpy array [T, V, 3]
            if isinstance(vertices, torch.Tensor):
                vertices_np = vertices.cpu().numpy()
            else:
                vertices_np = vertices
            
            # vertices shape should be [B*T, V, 3], reshape if needed
            if len(vertices_np.shape) == 3 and vertices_np.shape[0] == rst_len:
                # Already [T, V, 3]
                pass
            elif len(vertices_np.shape) == 4:
                # [B, T, V, 3] -> [T, V, 3] for first batch
                vertices_np = vertices_np[0]
            
            # Render video from meshes
            render_video_from_meshes(
                verts_list=vertices_np,
                faces=smpl_x.face,
                save_path=video_path,
                fps=fps
            )
            logger.info(f"Saved video to {video_path}")
            
        except Exception as e:
            logger.error(f"Failed to process text '{text}': {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"Inference complete. Results saved to {output_dir}")
