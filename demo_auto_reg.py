"""
Simple Demo Approach: Minimal modification to test.py
This shows how to add demo functionality to the existing test.py
"""

import os
import torch
import pytorch_lightning as pl
from mGPT.config import parse_args
from mGPT.models.build_model import build_model
from mGPT.data.build_data import build_data
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae

def demo_text_to_motion(model, datamodule, text_inputs, language='en', save_dir='demo_outputs'):
    """
    Generate sign language motions from text inputs
    Fixed version that handles retrieval properly

    Args:
        model: Trained MotionGPT model
        datamodule: Data module (for denormalization)
        text_inputs: List of text strings
        language: Target sign language ('en', 'zh', 'de')
        save_dir: Output directory

    Returns:
        List of generated motion results
    """
    import os
    import pickle

    # Ensure model is in eval mode
    model.eval()

    # Language prefixes
    lang_map = {'en': '<En>', 'zh': '<Zh>', 'de': '<De>'}
    lang_prefix = lang_map.get(language, '<En>')

    # Format texts with language prefix
    formatted_texts = [f"{lang_prefix} {text}" for text in text_inputs]

    # Create proper batch structure with all required fields
    batch = {
        'text': formatted_texts,
        'length': [None] * len(formatted_texts),  # Auto-determined
        # Add fields for retrieval (can be dummy values for demo)
        'name': [f'demo_{i}' for i in range(len(formatted_texts))],  # Sample names
        'word_tokens': ["how2sign"] * len(formatted_texts),  # Will be computed if needed
    }

    # Generate motions
    print(f"\nGenerating sign language for {len(text_inputs)} texts...")
    with torch.no_grad():
        try:
            # Try using the forward method (handles retrieval internally)
            outputs = model.forward(batch, task="t2m")
        except Exception as e:
            print(f"Warning: Forward method failed ({e}), trying alternative generation...")
            # Fallback: use val_t2m_forward which might handle missing fields better
            outputs = model.val_t2m_forward(batch)

    # Save outputs
    os.makedirs(save_dir, exist_ok=True)
    results = []

    for i, text in enumerate(text_inputs):
        result = {
            'text': text,
            'language': language,
            'feats': outputs['feats'][i].cpu().numpy() if 'feats' in outputs else outputs.get('m_rst', [None])[i].cpu().numpy(),
            'joints': outputs['joints'][i].cpu().numpy() if 'joints' in outputs else outputs.get('joints_rst', [None])[i].cpu().numpy(),
            'length': outputs['length'][i] if 'length' in outputs else outputs['feats'][i].shape[0],
        }

        # Save to pickle
        output_path = os.path.join(save_dir, f'demo_{i:03d}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)

        print(f"✓ '{text}' → {result['length']} frames → {output_path}")
        results.append(result)

    return results

def standalone_demo():
    """
    Complete standalone demo script
    """

    SEED_VALUE = 1234
    text = "你好世界"
    output_dir = "demo_outputs"
    language = "zh"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config
    cfg = parse_args(phase="test")  # parse config file
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg.DEVICE))

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Load data module (needed for model initialization)
    datamodule = build_data(cfg)

    # Build model
    model = build_model(cfg, datamodule)

    # Load checkpoint
    load_pretrained_vae(cfg, model, None)
    load_pretrained(cfg, model, None, phase="test")

    model.eval()
    model.to(device)

    # Generate motion
    print(f"Generating sign language for: '{text}'")
    result = demo_text_to_motion(
        model=model,
        datamodule=datamodule,
        text_inputs=[text],
        language=language,
        save_dir=output_dir
    )

    print(f"\n{'='*60}")
    print(f"✓ Success! Generated motion with {result[0]['length']} frames")
    print(f"✓ Output saved to {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    standalone_demo()