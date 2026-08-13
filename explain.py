#!/usr/bin/env python3
"""Application 1: semantic explanations at three granularities.

Reads your own face images and your own prompt vocabulary, and writes the
semantic signature of each image under a frozen FR model, plus a figure
with the three levels of explanation.

    python explain.py --images images --prompts prompts/signature_prompts.txt

See README.md for the expected folder layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from explfr import core
from explfr.models import (encode_text, get_device, load_adapter, load_clip,
                           load_fr_model, apply_adapter)
from explfr.registry import ADAPTER_DIR, DEFAULT_TARGET, REGISTRY, get_target

CASE_ROLES = ["reference", "genuine", "imposter", "morph"]


def parse_args():
    p = argparse.ArgumentParser(description="EXPL-FR semantic explanations")
    p.add_argument("--images", default="images", help="root image folder")
    p.add_argument("--prompts", default="prompts/signature_prompts.txt")
    p.add_argument("--target", default=DEFAULT_TARGET, choices=list(REGISTRY))
    p.add_argument("--fr-ckpt", required=True,
                   help="directory containing the FR model.pt")
    p.add_argument("--out", default="outputs", help="output directory")
    p.add_argument("--top-m", type=int, default=0,
                   help="keep only the M most detectable prompts (0 = keep all). "
                        "Needs at least ~200 images to be meaningful.")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def collect(images_root: Path):
    """identities/<name>/*.img  and  cases/<name>/{role}.img"""
    identities, cases = {}, {}
    id_root = images_root / "identities"
    if id_root.is_dir():
        for d in sorted(p for p in id_root.iterdir() if p.is_dir()):
            files = core.list_images(d)
            if files:
                identities[d.name] = files
    case_root = images_root / "cases"
    if case_root.is_dir():
        for d in sorted(p for p in case_root.iterdir() if p.is_dir()):
            roles = {}
            for f in core.list_images(d):
                if f.stem.lower() in CASE_ROLES:
                    roles[f.stem.lower()] = f
            if roles:
                cases[d.name] = roles
    if not identities and not cases:
        raise SystemExit(
            f"no images found under {images_root}. Expected "
            f"{images_root}/identities/<name>/*.jpg and/or "
            f"{images_root}/cases/<name>/reference.jpg (see README)."
        )
    return identities, cases


def main():
    args = parse_args()
    images_root = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    target = get_target(args.target)

    identities, cases = collect(images_root)
    paths, meta = [], []
    for name, files in identities.items():
        for f in files:
            paths.append(f)
            meta.append(("identity", name, f.stem))
    for name, roles in cases.items():
        for role in CASE_ROLES:
            if role in roles:
                paths.append(roles[role])
                meta.append(("case", name, role))
    print(f"[data] {len(paths)} images "
          f"({len(identities)} identities, {len(cases)} cases)")

    prompts, categories = core.read_prompts(args.prompts)
    print(f"[prompts] {len(prompts)} prompts in "
          f"{len(set(categories))} categories")

    print(f"[model] target={target.label}, device={device}")
    adapter = load_adapter(ADAPTER_DIR / target.adapter, device)
    fr_model, fr_tf = load_fr_model(args.fr_ckpt, target.fr_config, device)
    fr_emb = core.embed_images(paths, fr_model, fr_tf, device,
                               args.batch_size, "FR")

    text_emb = encode_text(prompts, device)
    anchors = core.anchors_from_text(text_emb, adapter, device)

    keep = np.arange(len(prompts))
    if args.top_m and args.top_m < len(prompts):
        if len(paths) < 100:
            print(f"[select] only {len(paths)} images; detectability needs a few "
                  "hundred to be stable, keeping all prompts")
        else:
            clip_model, clip_tf = load_clip(device)
            clip_emb = core.embed_images(
                paths, lambda x: clip_model.get_image_features(pixel_values=x),
                clip_tf, device, args.batch_size, "CLIP")
            mapped = apply_adapter(adapter, clip_emb, device)
            split = core.half_split(len(paths), seed=0)
            _, auc_fr = core.detectability(clip_emb, mapped, text_emb, split)
            keep = np.argsort(auc_fr)[::-1][:args.top_m]
            keep.sort()
            print(f"[select] kept {len(keep)} of {len(prompts)} prompts "
                  f"(AUC_F {auc_fr[keep].min():.3f}-{auc_fr[keep].max():.3f})")

    sig = core.signatures(fr_emb, anchors[keep])
    kept_prompts = [prompts[i] for i in keep]
    kept_categories = [categories[i] for i in keep]

    csv_path = out_dir / "signatures.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("level,group,item," + ",".join(
            f'"{p}"' for p in kept_prompts) + "\n")
        for (level, group, item), row in zip(meta, sig):
            fh.write(f"{level},{group},{item}," +
                     ",".join(f"{v:.6f}" for v in row) + "\n")
    print(f"[write] {csv_path}")

    fig_path = out_dir / "explanations.pdf"
    plot(fig_path, sig, meta, kept_categories, identities, cases)
    print(f"[write] {fig_path}")


def plot(path, sig, meta, categories, identities, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(np.array(categories), kind="stable")
    cats = np.array(categories)[order]
    bounds = [0] + [i for i in range(1, len(cats)) if cats[i] != cats[i - 1]] + [len(cats)]

    rows = []                                   # (title, values, is_diff)
    for name in identities:
        idx = [i for i, m in enumerate(meta) if m[0] == "identity" and m[1] == name]
        rows.append((f"identity: {name}", sig[idx].mean(0), False))
    for name, roles in cases.items():
        pos = {m[2]: i for i, m in enumerate(meta) if m[0] == "case" and m[1] == name}
        for role in CASE_ROLES:
            if role in pos:
                rows.append((f"{name}: {role}", sig[pos[role]], False))
        if "reference" in pos:
            for role in ("genuine", "imposter", "morph"):
                if role in pos:
                    rows.append((f"{name}: reference - {role}",
                                 sig[pos["reference"]] - sig[pos[role]], True))
    if not rows:
        return

    # Each signature row is shown relative to its own mean. Every anchor
    # shares a common component (the VLM's modality gap, mapped into FR
    # space), which shifts a whole row up or down without saying anything
    # about concepts; centring removes it and leaves the relative pattern.
    # Difference rows already cancel it, so they are left as they are.
    rows = [(t, (v - v.mean()) if not d else v, d) for t, v, d in rows]

    # shared scale within each kind, so rows are comparable at a glance
    sig_vals = [v for _, v, d in rows if not d]
    dif_vals = [v for _, v, d in rows if d]
    sig_lim = float(np.abs(np.concatenate(sig_vals)).max()) if sig_vals else 1.0
    dif_lim = float(np.abs(np.concatenate(dif_vals)).max()) if dif_vals else 1.0

    h = max(2.0, 0.9 * len(rows))
    fig, axes = plt.subplots(len(rows), 1, figsize=(14, h), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    palette = plt.get_cmap("tab20")
    for ax, (title, values, is_diff) in zip(axes, rows):
        v = values[order]
        colors = [palette(bi % 20) for bi in range(len(bounds) - 1)
                  for _ in range(bounds[bi + 1] - bounds[bi])]
        ax.bar(np.arange(len(v)), v, color=colors, width=0.9,
               edgecolor="black", linewidth=0.2)
        ax.axhline(0, color="black", lw=0.6)
        for b in bounds[1:-1]:
            ax.axvline(b - 0.5, color="0.85", lw=0.8, zorder=0)
        lim = dif_lim if is_diff else sig_lim
        ax.set_ylim(-1.05 * lim, 1.05 * lim)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_ylabel(title, fontsize=7, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=6)
        ax.set_xlim(-0.5, len(v) - 0.5)
    axes[-1].set_xticks([(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1)])
    axes[-1].set_xticklabels([cats[bounds[i]] for i in range(len(bounds) - 1)],
                             rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except (EnvironmentError, FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}")
