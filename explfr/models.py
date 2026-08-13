"""Frozen encoders and the trained CLIP -> FR adapter.

Nothing here is trained. The VLM and the FR model are loaded frozen from
their public releases; the adapter is the only EXPL-FR component and its
weights ship with this repository.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------

def build_adapter(depth: int = 4, dim: int = 512) -> nn.Sequential:
    """The 4-layer MLP of the paper: (Linear, BatchNorm, GELU) x3, Linear."""
    layers = []
    for _ in range(depth - 1):
        layers += [nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), nn.GELU()]
    layers.append(nn.Linear(dim, dim, bias=True))
    return nn.Sequential(*layers)


def load_adapter(ckpt_path, device, depth: int = 4, dim: int = 512) -> nn.Sequential:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"adapter checkpoint not found: {ckpt_path}\n"
            "Download the weights and place them in adapter/ (see README)."
        )
    adapter = build_adapter(depth, dim)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k[len("net."):] if k.startswith("net.") else k: v for k, v in state.items()}
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    if len(missing) == len(adapter.state_dict()):
        raise RuntimeError(f"no weights matched in {ckpt_path}; wrong checkpoint?")
    adapter = adapter.to(device).eval()
    for p in adapter.parameters():
        p.requires_grad = False
    return adapter


@torch.no_grad()
def apply_adapter(adapter, x: np.ndarray, device, batch: int = 4096) -> np.ndarray:
    """(N, d) -> (N, d) unit-norm. Used for text (anchors) and for images."""
    out = []
    for i in range(0, len(x), batch):
        t = torch.from_numpy(np.ascontiguousarray(x[i:i + batch])).float().to(device)
        out.append(F.normalize(adapter(t), dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------------------
# vision-language model (frozen)
# ---------------------------------------------------------------------------

def load_clip(device):
    """Returns (model, image_transform). Weights are fetched by transformers."""
    from transformers import CLIPModel
    from torchvision import transforms as T

    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.Lambda(lambda im: im.convert("RGB")),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    return model, transform


@torch.no_grad()
def encode_text(prompts, device, batch: int = 256) -> np.ndarray:
    """Unit-norm CLIP text embeddings, (K, d) float32."""
    from transformers import CLIPModel, CLIPTokenizer
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], padding=True, truncation=True,
                  return_tensors="pt").to(device)
        emb = model.get_text_features(**enc)
        out.append(F.normalize(emb, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------------------
# face recognition model (frozen, external)
# ---------------------------------------------------------------------------

def load_fr_model(ckpt_dir, config, device):
    """Load a CVLface face recognition model. Returns (model, transform).

    ckpt_dir : directory containing model.pt
    config   : config path relative to the CVLface run_v1 directory,
               e.g. models/vit/configs/v1_base.yaml
    Requires the environment variable CVLFACE_RUN_V1 to point at
    <CVLface>/cvlface/research/recognition/code/run_v1 (see README).
    """
    import sys
    from torchvision import transforms as T

    run_v1 = os.environ.get("CVLFACE_RUN_V1", "")
    if not run_v1 or not Path(run_v1).is_dir():
        raise EnvironmentError(
            "CVLFACE_RUN_V1 is not set to a valid directory.\n"
            "Clone https://github.com/mk-minchul/CVLface and export\n"
            "  export CVLFACE_RUN_V1=<CVLface>/cvlface/research/recognition/code/run_v1"
        )

    # CVLface imports mxnet, which still uses numpy aliases removed in 1.24
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for name, alias in (("bool", bool), ("object", object)):
            if not hasattr(np, name):
                setattr(np, name, alias)

    import pyrootutils
    root = pyrootutils.setup_root(search_from=run_v1, indicator=["__root__.txt"],
                                  pythonpath=True, dotenv=True)
    sys.path.insert(0, run_v1)
    sys.path.insert(0, str(root))

    from general_utils.config_utils import load_config
    from models import get_model

    config_path = os.path.join(run_v1, config)
    cfg = load_config(config_path)
    cfg.yaml_path = config_path
    model = get_model(cfg, task="run_v1")

    model_pt = Path(ckpt_dir) / "model.pt"
    if not model_pt.is_file():
        raise FileNotFoundError(f"FR checkpoint not found: {model_pt}")
    state = torch.load(model_pt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k[len("model."):] if k.startswith("model.") else k: v
             for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    steps = [T.Resize((112, 112)), T.Lambda(lambda im: im.convert("RGB")), T.ToTensor()]
    if getattr(cfg, "color_space", "bgr").lower() == "bgr":
        steps.append(T.Lambda(lambda x: x[[2, 1, 0], :, :]))
    steps.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    return model, T.Compose(steps)
