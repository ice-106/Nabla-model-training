import torch
from mGPT.utils.misc import neq_load_customized


_VOCAB_PARAMETER_SUFFIXES = (
    "final_logits_bias",
    "model.shared.weight",
    "model.encoder.embed_tokens.weight",
    "model.decoder.embed_tokens.weight",
    "lm_head.weight",
)


def _vocab_size(tensor, key):
    return tensor.shape[-1] if key.endswith("final_logits_bias") else tensor.shape[0]


def _validate_checkpoint_vocab(model, state_dict):
    """Fail clearly when an experiment selects the wrong historical LM variant."""
    model_state = model.state_dict()
    mismatches = []
    for key, checkpoint_tensor in state_dict.items():
        if not key.endswith(_VOCAB_PARAMETER_SUFFIXES) or key not in model_state:
            continue
        model_tensor = model_state[key]
        if checkpoint_tensor.shape != model_tensor.shape:
            mismatches.append(
                (key, _vocab_size(checkpoint_tensor, key), _vocab_size(model_tensor, key))
            )

    if not mismatches:
        return

    checkpoint_sizes = sorted({item[1] for item in mismatches})
    model_sizes = sorted({item[2] for item in mismatches})
    details = ", ".join(
        f"{key}: checkpoint={checkpoint_size}, model={model_size}"
        for key, checkpoint_size, model_size in mismatches
    )
    raise RuntimeError(
        "Checkpoint vocabulary is incompatible with the selected LM variant. "
        f"Checkpoint vocab size(s): {checkpoint_sizes}; model vocab size(s): {model_sizes}. "
        "Use lm.mbart_h2s_csl_phoenix for the historical three-language checkpoint "
        "or lm.mbart_h2s_csl_phoenix_thai for Thai/four-language checkpoints. "
        f"Mismatched tensors: {details}"
    )


def load_pretrained(cfg, model, logger=None, phase="train"):    
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
    
    if logger is not None:
        logger.info(f"Loading pretrain model from {ckpt_path}")
    
    # [MODIFIED] To be able to load from given checkpoint
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]


    _validate_checkpoint_vocab(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    return model


def load_pretrained_vae(cfg, model, logger=None):
    # [MODIFIED] To be able to load from given checkpoint
    state_dict = torch.load(cfg.TRAIN.PRETRAINED_VAE,
                            map_location="cpu", weights_only=False)['state_dict']
    if logger is not None:
        logger.info(f"Loading pretrain vae from {cfg.TRAIN.PRETRAINED_VAE}")
        
    # Extract encoder/decoder
    from collections import OrderedDict
    vae_dict = OrderedDict()
    hand_vae_dict = OrderedDict()
    rhand_vae_dict = OrderedDict()
    for k, v in state_dict.items():
        if "motion_vae" in k:
            name = k.replace("motion_vae.", "")
            vae_dict[name] = v
        elif "rhand_vae" in k:
            name = k.replace("rhand_vae.", "")
            rhand_vae_dict[name] = v
        elif "hand_vae" in k:
            name = k.replace("hand_vae.", "")
            hand_vae_dict[name] = v
        elif "vae" in k:
            name = k.replace("vae.", "")
            vae_dict[name] = v
    
    if hasattr(model, 'rhand_vae'):
        print('load rhand vae...')
        neq_load_customized(model.rhand_vae, rhand_vae_dict, verbose=True)
    if hasattr(model, 'hand_vae'):
        print('load hand vae...')
        neq_load_customized(model.hand_vae, hand_vae_dict, verbose=True)
    if hasattr(model, 'vae'):
        print('load vae...')
        # model.vae.load_state_dict(vae_dict, strict=True)
        neq_load_customized(model.vae, vae_dict, verbose=True)
    else:
        # model.motion_vae.load_state_dict(vae_dict, strict=True)
        neq_load_customized(model.motion_vae, vae_dict, verbose=True)
    
    return model
