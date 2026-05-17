# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important note
- Please use conda environment 'pytorchmacos' to run code.
- Any code you generate should be locate in agentScripts folder or /tmp/ if you prefer. Note: `agentScripts/` is gitignored — files there are local scratch and never enter the repo. Use `scripts/` for utilities that should be checked in (invoke as `python -m scripts.<name>` from the repo root).

## Project Overview

SOKE (Signs as Tokens) - official implementation of the ICCV 2025 paper "Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator". Generates 3D sign language avatars from text using a pretrained language model with a two-stage training pipeline built on PyTorch Lightning.

## Commands

### Environment Setup
```bash
conda create python=3.10 --name soke
conda activate soke
pip install -e .                    # development install (loose top-level deps from pyproject.toml)
# or, for an exact reproduction of the verified env:
pip install -r requirements.lock
sh prepare/download_t2m_evaluators.sh
sh prepare/prepare_t5.sh
```

### Stage 1: Decoupled Tokenizer (VQVae)
```bash
# Train tokenizer
python -m train --cfg configs/deto.yaml --nodebug

# Inference/evaluate tokenizer
python -m test --cfg configs/deto.yaml --nodebug
```

### Stage 2: Autoregressive Language Model (mBart)
```bash
# Generate motion codes from trained tokenizer (must run before LM training)
python -m scripts.get_motion_code --cfg configs/soke.yaml --nodebug

# Train LM (update PRETRAINED_VAE path in soke.yaml first)
python -m train --cfg configs/soke.yaml --nodebug

# Inference
python -m test --cfg configs/soke.yaml --task t2m
```

### Visualization
```bash
python -m scripts.vis_mesh --cfg=configs/soke.yaml --demo_dataset=csl
python -m scripts.vis_blender
```

### Useful CLI flags
- `--nodebug` - required for real training (without it, runs in debug mode with wandb offline and 1-step validation)
- `--batch_size N` - override batch size
- `--device 0 1` - specify GPU indices
- `--task t2m` - set evaluation task type
- `--use_gpus "0,1,2,3"` - set CUDA_VISIBLE_DEVICES

## Architecture

### Two-Stage Pipeline

**Stage 1 (VAE/Tokenizer)** - Discretizes continuous sign motion into tokens:
- Three separate VQVae models: body (re96 config, 96 codebook entries), left hand, right hand (hand192 config, 192 entries each)
- Motion is split: joints 0-29 + 120+ → body, 30-74 → left hand, 75-119 → right hand
- Core: `mGPT/archs/mgpt_vq.py` (Encoder → EMA-Reset Quantizer → Decoder)

**Stage 2 (Language Model)** - Text-to-motion token generation:
- mBart-large-cc25 encoder-decoder with extended motion token embeddings
- Multi-head decoder (`mGPT/archs/lm_multihead.py`) predicts body, left hand, and right hand tokens simultaneously
- Supports multilingual: en_XX→en_ASL, zh_CN→zh_CSL, de_DE→de_DGS, th_TH→th_THS
- Core: `mGPT/archs/mgpt_mbart.py`

### Key Module Layout

- `mGPT/models/mgpt.py` - Main MotionGPT LightningModule orchestrating both stages. Routes to different forward/train/test logic based on `TRAIN.STAGE` (vae, lm_pretrain, lm_instruct)
- `mGPT/archs/` - Model architectures (VQVae, mBart LM, multi-head decoders)
- `mGPT/data/H2S.py` - Primary data module supporting How2Sign (ASL), CSL-Daily (Chinese), Phoenix-2014T (German). Dataset class varies by stage (VQ, CB, Token, M2T)
- `mGPT/losses/mgpt.py` - Loss functions: recons_feature (SmoothL1), vq_commit, gpt_loss (CrossEntropy)
- `mGPT/metrics/` - MRMetrics (APE, AVE, FID) for VAE, TM2TMetrics for LM
- `mGPT/callback.py` - Checkpoint callback saving last.ckpt every epoch
- `mGPT/config.py` - Config loading: merges default.yaml + experiment yaml + subconfigs from `configs/*/` + assets.yaml via OmegaConf

### Config System

Configs use OmegaConf with hierarchical merging. `configs/default.yaml` is base, experiment configs (soke.yaml, deto.yaml) override it, and subfolder configs (`configs/lm/`, `configs/vq/`, `configs/evaluator/`) are auto-loaded as nested keys (e.g., `${lm.mbart_h2s_csl_phoenix}`, `${vq.re96}`). Asset paths live in `configs/assets.yaml`.

### Data Dependencies

External data expected at paths relative to project root (configured in yaml files):
- `../data/How2Sign/`, `../data/CSL-Daily/`, `../data/Phoenix_2014T/` - datasets with SMPL-X poses
- `../pretrained/tokenizer.ckpt` - pretrained tokenizer checkpoint
- `deps/smpl_models/` - SMPL/SMPL-X human models
- `deps/mbart-h2s-csl-phoenix/` - pretrained mBart weights

### soke.yaml Checklist (from config comments)

When modifying soke.yaml for a new experiment:
1. Change NAME (creates new experiment folder)
2. Set DEBUG to False and `--nodebug` flag
3. Verify DATASET_NAME (h2s, csl, or phoenix)
4. Verify RESUME path if continuing training
5. Verify WANDB project name
