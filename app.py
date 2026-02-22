from mGPT.utils.render_utils import render_video_from_meshes
from mGPT.utils.human_models import smpl_x
from mGPT.utils.logger import create_logger
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae
from mGPT.models.build_model import build_model
from mGPT.data.inference_data import InferenceDataModule
from mGPT.config import get_module_config, instantiate_from_config
from os.path import join as pjoin
from omegaconf import OmegaConf
from argparse import ArgumentParser
import pytorch_lightning as pl
import gradio as gr
import pickle
import torch.nn.functional as F
import torch
import tempfile
import logging
import os
# Set headless rendering before any other imports
os.environ["PYOPENGL_PLATFORM"] = "egl"


# ── Global state (loaded once at startup) ──────────────────────────
MODEL = None
CFG = None
DATAMODULE = None
LOGGER = None


def load_model(cfg_path: str, cfg_assets: str, checkpoint: str = None):
    """Load config, datamodule, and model – mirrors test.py setup."""
    global MODEL, CFG, DATAMODULE, LOGGER

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_assets_obj = OmegaConf.load(cfg_assets)
    cfg_base = OmegaConf.load(
        pjoin(cfg_assets_obj.CONFIG_FOLDER, "default.yaml"))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets_obj.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets_obj)

    # Force test-mode settings
    cfg.DEBUG = False
    cfg.FOLDER = cfg.get("FOLDER", "./results")
    cfg.TEST.SPLIT = "test"

    # Override checkpoint if provided via CLI
    if checkpoint:
        cfg.TEST.CHECKPOINTS = checkpoint

    LOGGER = create_logger(cfg, phase="test")
    pl.seed_everything(cfg.SEED_VALUE)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    DATAMODULE = InferenceDataModule(cfg)
    MODEL = build_model(cfg, DATAMODULE)

    # Load pretrained VAE
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, MODEL, LOGGER)

    # Load checkpoint
    if not cfg.TEST.CHECKPOINTS:
        ckpt_folder = os.path.join(
            cfg.FOLDER_EXP.replace("results", "experiments"), "checkpoints"
        )
        cfg.TEST.CHECKPOINTS = os.path.join(ckpt_folder, "last.ckpt")
    load_pretrained(cfg, MODEL, LOGGER, phase="test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = MODEL.to(device)
    MODEL.eval()

    CFG = cfg
    LOGGER.info("Model loaded and ready for inference.")


# ── Inference function called by Gradio ─────────────────────────────
def generate_sign(text: str, src: str, fps: int = 20):
    """Generate a sign-language video from *text* and return the video path."""
    if MODEL is None:
        return None

    if not text or not text.strip():
        raise gr.Error("Please enter some text.")

    device = next(MODEL.parameters()).device
    output_dir = tempfile.mkdtemp(prefix="nabla_infer_")
    feats2joints = MODEL.feats2joints

    LOGGER.info(f"Generating sign for: '{text}' | src={src}")

    with torch.no_grad():
        gen_results = MODEL.lm.generate_conditional(
            [text],
            lengths=None,
            stage="test",
            tasks=None,
            src=[src],
            name=["gradio_infer"],
        )

    outputs_tokens = gen_results["outputs_tokens"]
    outputs_tokens_hand = gen_results.get("outputs_tokens_hand")
    outputs_tokens_rhand = gen_results.get("outputs_tokens_rhand")

    # Determine max length for padding
    max_len = max(map(len, outputs_tokens))
    if outputs_tokens_hand is not None:
        max_len = max(max_len, max(map(len, outputs_tokens_hand)))
    if outputs_tokens_rhand is not None:
        max_len = max(max_len, max(map(len, outputs_tokens_rhand)))
    max_len *= 4  # upsample factor

    C = 133  # body(30) + lhand(45) + rhand(45) + face(13)
    feats_rst = torch.zeros(1, max_len, C, device=device)
    rst_len = max_len

    # ── Decode body tokens ──────────────────────────────────────
    outputs_tokens[0] = torch.clamp(
        outputs_tokens[0], 0, MODEL.vae.code_num - 1)
    if len(outputs_tokens[0]) > 1:
        motion = MODEL.vae.decode(outputs_tokens[0])
        rst_len = motion.shape[1]
        motion = F.pad(motion, (0, 0, 0, max_len -
                       motion.shape[1]), mode="replicate")
    else:
        motion = torch.zeros((1, max_len, MODEL.vae.nfeats), device=device)
        rst_len = 1

    feats_rst[0:1, :, :30] = motion[..., :30]
    feats_rst[0:1, :, -13:] = motion[..., 30:43]

    # ── Decode left-hand tokens ─────────────────────────────────
    if outputs_tokens_hand is not None and hasattr(MODEL, "hand_vae"):
        outputs_tokens_hand[0] = torch.clamp(
            outputs_tokens_hand[0], 0, MODEL.hand_vae.code_num - 1
        )
        if len(outputs_tokens_hand[0]) > 1:
            motion_hand = MODEL.hand_vae.decode(outputs_tokens_hand[0])
            rst_len = max(rst_len, motion_hand.shape[1])
            motion_hand = F.pad(
                motion_hand, (0, 0, 0, max_len - motion_hand.shape[1]), mode="replicate"
            )
        else:
            motion_hand = torch.zeros(
                (1, max_len, MODEL.hand_vae.nfeats), device=device)
        feats_rst[0:1, :, 30: 30 + MODEL.hand_vae.nfeats] = motion_hand

    # ── Decode right-hand tokens ────────────────────────────────
    if outputs_tokens_rhand is not None and hasattr(MODEL, "rhand_vae"):
        outputs_tokens_rhand[0] = torch.clamp(
            outputs_tokens_rhand[0], 0, MODEL.rhand_vae.code_num - 1
        )
        if len(outputs_tokens_rhand[0]) > 1:
            motion_rhand = MODEL.rhand_vae.decode(outputs_tokens_rhand[0])
            rst_len = max(rst_len, motion_rhand.shape[1])
            motion_rhand = F.pad(
                motion_rhand, (0, 0, 0, max_len - motion_rhand.shape[1]), mode="replicate"
            )
        else:
            motion_rhand = torch.zeros(
                (1, max_len, MODEL.rhand_vae.nfeats), device=device)
        feats_rst[0:1, :, 75: 75 + MODEL.rhand_vae.nfeats] = motion_rhand

    # Trim to actual length
    feats_rst = feats_rst[:, :rst_len, :]

    # ── Convert to SMPL-X vertices ──────────────────────────────
    ret = feats2joints(feats_rst)
    vertices = ret[0] if isinstance(ret, tuple) else ret

    vertices_np = vertices.cpu().numpy() if isinstance(
        vertices, torch.Tensor) else vertices
    if len(vertices_np.shape) == 4:
        vertices_np = vertices_np[0]

    # ── Render video ────────────────────────────────────────────
    video_path = os.path.join(output_dir, "output.mp4")
    render_video_from_meshes(
        verts_list=vertices_np,
        faces=smpl_x.face,
        save_path=video_path,
        fps=fps,
    )
    LOGGER.info(f"Video saved to {video_path}")
    return video_path


# ── Gradio UI ───────────────────────────────────────────────────────
def build_demo():
    with gr.Blocks(title="Nabla – Text2Sign Generator") as demo:
        gr.Markdown("## Nabla – Text to Sign Language Generator")
        gr.Markdown(
            "Enter text and choose the source language/dataset, "
            "then click **Generate Sign** to produce a sign-language video."
        )

        with gr.Row():
            with gr.Column(scale=3):
                textbox = gr.Textbox(
                    lines=2,
                    label="Input Text",
                    placeholder="Enter your sentence here...",
                )
                with gr.Row():
                    src_dropdown = gr.Dropdown(
                        choices=["how2sign", "csl", "phoenix", "thai"],
                        value="how2sign",
                        label="Source Language / Dataset",
                    )
                    fps_slider = gr.Slider(
                        minimum=10,
                        maximum=60,
                        value=20,
                        step=5,
                        label="Video FPS",
                    )
                generate_btn = gr.Button("Generate Sign", variant="primary")

            with gr.Column(scale=4):
                video_output = gr.Video(
                    label="Generated Sign Video", height=480)

        generate_btn.click(
            fn=generate_sign,
            inputs=[textbox, src_dropdown, fps_slider],
            outputs=video_output,
        )

        gr.Examples(
            examples=[
                ["hello", "how2sign"],
                ["thank you", "how2sign"],
                ["nice to meet you", "csl"],
            ],
            inputs=[textbox, src_dropdown],
            label="Try these examples",
        )

    return demo


# ── Entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = ArgumentParser(description="Nabla Text2Sign Gradio Demo")
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to experiment config YAML (e.g. configs/soke-full-dataset.yaml)",
    )
    parser.add_argument(
        "--cfg_assets",
        type=str,
        default="./configs/assets.yaml",
        help="Path to assets config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path (optional, otherwise uses TEST.CHECKPOINTS from cfg)",
    )
    parser.add_argument(
        "--use_gpus",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES (default: '0')",
    )
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port number")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpus

    # Load model once at startup
    load_model(args.cfg, args.cfg_assets, checkpoint=args.checkpoint)

    # Launch Gradio
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
