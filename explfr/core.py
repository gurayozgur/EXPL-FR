"""Embedding extraction, semantic signatures, and the audit statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def l2n(x: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def list_images(folder) -> List[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


# ---------------------------------------------------------------------------
# prompt files
# ---------------------------------------------------------------------------

def read_prompts(path) -> Tuple[List[str], List[str]]:
    """Signature vocabulary. One prompt per line, optionally 'category<TAB>prompt'.
    Blank lines and lines starting with # are ignored."""
    prompts, categories = [], []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            cat, prompt = line.split("\t", 1)
        else:
            cat, prompt = "misc", line
        prompts.append(prompt.strip())
        categories.append(cat.strip() or "misc")
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts, categories


def read_axes(path) -> Dict[str, List[str]]:
    """Audit axes. '[axis name]' starts a block; following lines are its
    ordered prompts (low end first, high end last). At least 2 per axis."""
    axes: Dict[str, List[str]] = {}
    current = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            axes[current] = []
        elif current is not None:
            axes[current].append(line)
    axes = {k: v for k, v in axes.items() if v}
    bad = [k for k, v in axes.items() if len(v) < 2]
    if bad:
        raise ValueError(f"axes need at least 2 ordered prompts each: {bad}")
    if not axes:
        raise ValueError(f"no axes found in {path}")
    return axes


# ---------------------------------------------------------------------------
# image embeddings
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_images(paths: Sequence[Path], encode, transform, device,
                 batch_size: int = 64, desc: str = "embed") -> np.ndarray:
    """PIL -> transform -> encode -> (N, D) unit-norm float32."""
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            with Image.open(paths[i]) as im:
                return transform(im)

    loader = DataLoader(_DS(), batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    for n, batch in enumerate(loader, 1):
        emb = encode(batch.to(device))
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
        out.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy())
        print(f"\r  {desc}: {min(n * batch_size, len(paths))}/{len(paths)}",
              end="", flush=True)
    print()
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------------------
# signatures  (Eq. 3 and Eq. 4 of the paper)
# ---------------------------------------------------------------------------

def anchors_from_text(text_emb: np.ndarray, adapter, device) -> np.ndarray:
    """p_k = normalize(g(t_k)): FR-space anchor per prompt, (K, D)."""
    from .models import apply_adapter
    return apply_adapter(adapter, text_emb, device)


def signatures(fr_emb: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """s_k(x) = <f(x), p_k>, shape (N, K)."""
    return fr_emb @ anchors.T


# ---------------------------------------------------------------------------
# label-free detectability, used to select the semantic signature
# ---------------------------------------------------------------------------

def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def detectability(clip_emb: np.ndarray, mapped_emb: np.ndarray,
                  text_emb: np.ndarray, split: np.ndarray,
                  q: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    """Label-free detectability per prompt (paper Sec. 3.3).

    The VLM's centred grounding score pseudo-labels each prompt's extremes;
    a mean-difference detector is fitted on one half and scored on the other,
    once in VLM space (AUC_V) and once in adapter-mapped FR space (AUC_F).

    Returns (auc_vlm, auc_fr), each (K,). Selection uses AUC_F.
    """
    t = l2n(text_emb - text_emb.mean(0)[None, :])
    r = clip_emb @ t.T                                   # (N, K) grounding scores
    a = np.flatnonzero(split)
    b = np.flatnonzero(~split)
    ka = max(10, int(q * len(a)))
    kb = max(10, int(q * len(b)))
    K = t.shape[0]
    auc_v = np.zeros(K)
    auc_f = np.zeros(K)
    for k in range(K):
        oa = np.argsort(r[a, k])
        hi, lo = a[oa[-ka:]], a[oa[:ka]]
        w_v = clip_emb[hi].mean(0) - clip_emb[lo].mean(0)
        w_f = mapped_emb[hi].mean(0) - mapped_emb[lo].mean(0)
        ob = np.argsort(r[b, k])
        keep = np.concatenate([b[ob[:kb]], b[ob[-kb:]]])
        lab = np.zeros(len(keep), dtype=bool)
        lab[kb:] = True
        auc_v[k] = _auc(clip_emb[keep] @ w_v, lab)
        auc_f[k] = _auc(mapped_emb[keep] @ w_f, lab)
    return auc_v, auc_f


def half_split(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=n // 2, replace=False)] = True
    return mask


# ---------------------------------------------------------------------------
# audit: attribute axes  (paper Sec. 3.5, settings 2 and 3)
# ---------------------------------------------------------------------------

def axis_prompt_only(axis_text_emb: np.ndarray, adapter, device) -> np.ndarray:
    """Setting (3), ours. First principal direction of the adapter-mapped
    ordered prompt set, oriented from the first prompt to the last.
    No images are used to build this axis."""
    from .models import apply_adapter
    P = apply_adapter(adapter, axis_text_emb, device)
    d = P - P.mean(0)
    if len(P) == 2:
        axis = d[1] - d[0]
    else:
        _, _, Vt = np.linalg.svd(d, full_matrices=False)
        axis = Vt[0]
        if (d[-1] - d[0]) @ axis < 0:
            axis = -axis
    return l2n(axis[None, :])[0].astype(np.float32)


def axis_vlm_proxy(fr_emb: np.ndarray, clip_emb: np.ndarray,
                   axis_text_emb: np.ndarray, text_mu: np.ndarray,
                   q: float = 0.20) -> np.ndarray:
    """Setting (2), VLM-proxy. The VLM ranks the unlabeled pool with the
    axis end-points; the axis is the FR-space mean difference between the
    top-q and bottom-q images. Uses images, but no labels."""
    t = l2n(axis_text_emb - text_mu[None, :])
    score = clip_emb @ t[-1] - clip_emb @ t[0]           # last end vs first end
    n = len(fr_emb)
    m = max(1, int(round(q * n)))
    order = np.argsort(score)
    d = fr_emb[order[-m:]].mean(0) - fr_emb[order[:m]].mean(0)
    return l2n(d[None, :])[0].astype(np.float32)


def axis_labeled(fr_emb: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Setting (1), prior practice. Mean difference between labelled groups."""
    d = fr_emb[pos].mean(0) - fr_emb[~pos].mean(0)
    return l2n(d[None, :])[0].astype(np.float32)


# ---------------------------------------------------------------------------
# audit: dependence statistic
# ---------------------------------------------------------------------------

def _pair_dists(a: np.ndarray, b: np.ndarray, n_pairs: int, rng, same: bool):
    ia = rng.integers(0, len(a), size=n_pairs)
    ib = rng.integers(0, len(b), size=n_pairs)
    if same:
        clash = ia == ib
        ib[clash] = (ib[clash] + 1) % len(b)
    return 1.0 - np.einsum("nd,nd->n", a[ia], b[ib])


def dependence(fr_emb: np.ndarray, axis: np.ndarray, frac: float = 0.25,
               n_pairs: int = 20000, seed: int = 0) -> float:
    """How strongly a concept structures the FR space (paper Sec. 3.5).

    The pool is split into the top and bottom `frac` by projection onto the
    axis; the statistic is the two-sample KS distance between within-group
    and across-group FR distance distributions, averaged over the groups.
    Higher means the concept structures the space more.
    """
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(seed)
    v = fr_emb @ axis
    n = len(v)
    m = max(2, int(round(frac * n)))
    order = np.argsort(v)
    groups = [fr_emb[order[:m]], fr_emb[order[-m:]]]
    if min(len(g) for g in groups) < 10:
        return float("nan")
    stats = []
    for i in (0, 1):
        intra = _pair_dists(groups[i], groups[i], n_pairs, rng, same=True)
        inter = _pair_dists(groups[i], groups[1 - i], n_pairs, rng, same=False)
        stats.append(ks_2samp(intra, inter).statistic)
    return float(np.mean(stats))


def sensitivity(fr_a: np.ndarray, fr_b: np.ndarray, axis: np.ndarray) -> float:
    """Matched-axis sensitivity |<f(a) - f(b), axis>|, averaged over pairs."""
    return float(np.abs((fr_a - fr_b) @ axis).mean())
