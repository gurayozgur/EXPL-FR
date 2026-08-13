#!/usr/bin/env python3
"""Application 2: attribute-level auditing of a frozen FR model.

Measures, for each named attribute axis, how strongly that concept
structures the FR embedding space of an unlabeled image pool. Two
label-free settings are provided:

  ours   the axis is built from written prompts alone (paper setting 3)
  proxy  the VLM ranks the pool to build the axis  (paper setting 2)

    python audit.py --images images/pool --axes prompts/audit_axes.txt \
                    --fr-ckpt /path/to/fr_checkpoint

See README.md for the expected folder layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from explfr import core
from explfr.models import (encode_text, get_device, load_adapter, load_clip,
                           load_fr_model)
from explfr.registry import ADAPTER_DIR, DEFAULT_TARGET, REGISTRY, get_target


def parse_args():
    p = argparse.ArgumentParser(description="EXPL-FR attribute auditing")
    p.add_argument("--images", default="images/pool",
                   help="flat folder of unlabeled, aligned face images")
    p.add_argument("--axes", default="prompts/audit_axes.txt")
    p.add_argument("--target", default=DEFAULT_TARGET, choices=list(REGISTRY))
    p.add_argument("--fr-ckpt", required=True,
                   help="directory containing the FR model.pt")
    p.add_argument("--setting", default="ours", choices=["ours", "proxy", "both"])
    p.add_argument("--out", default="outputs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--frac", type=float, default=0.25,
                   help="top/bottom fraction of the pool forming the two groups")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    target = get_target(args.target)

    paths = core.list_images(args.images)
    if len(paths) < 50:
        raise SystemExit(
            f"found {len(paths)} images in {args.images}; the dependence "
            "statistic needs a few hundred to be meaningful (see README)."
        )
    axes_prompts = core.read_axes(args.axes)
    print(f"[data] {len(paths)} images, {len(axes_prompts)} axes: "
          f"{', '.join(axes_prompts)}")

    print(f"[model] target={target.label}, device={device}")
    adapter = load_adapter(ADAPTER_DIR / target.adapter, device)
    fr_model, fr_tf = load_fr_model(args.fr_ckpt, target.fr_config, device)
    fr_emb = core.embed_images(paths, fr_model, fr_tf, device,
                               args.batch_size, "FR")

    need_clip = args.setting in ("proxy", "both")
    clip_emb = None
    if need_clip:
        clip_model, clip_tf = load_clip(device)
        clip_emb = core.embed_images(
            paths, lambda x: clip_model.get_image_features(pixel_values=x),
            clip_tf, device, args.batch_size, "CLIP")

    # one text embedding per axis end-point; the vocabulary mean centres them
    flat, spans = [], {}
    for name, prompts in axes_prompts.items():
        spans[name] = (len(flat), len(flat) + len(prompts))
        flat.extend(prompts)
    text_emb = encode_text(flat, device)
    text_mu = text_emb.mean(0)

    settings = ["ours", "proxy"] if args.setting == "both" else [args.setting]
    rows = []
    for name, (i, j) in spans.items():
        axis_text = text_emb[i:j]
        for setting in settings:
            if setting == "ours":
                axis = core.axis_prompt_only(axis_text, adapter, device)
            else:
                axis = core.axis_vlm_proxy(fr_emb, clip_emb, axis_text, text_mu)
            dep = core.dependence(fr_emb, axis, frac=args.frac)
            rows.append((name, setting, len(axis_text), dep))
            print(f"  {name:<24s} {setting:<6s} dependence = {dep:.4f}")

    csv_path = out_dir / "audit.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("axis,setting,n_prompts,dependence\n")
        for name, setting, n, dep in rows:
            fh.write(f"{name},{setting},{n},{dep:.6f}\n")
    print(f"[write] {csv_path}")

    for setting in settings:
        sub = [(n, d) for n, s, _, d in rows if s == setting and not np.isnan(d)]
        if sub:
            sub.sort(key=lambda t: -t[1])
            ranking = " > ".join(n for n, _ in sub)
            print(f"[{setting}] most to least structuring: {ranking}")

    print("\nHigher dependence means the FR space is structured more strongly "
          "along that concept.\nCompare axes within one model, or the same axis "
          "across models. Absolute values\nare not calibrated between different "
          "image pools.")


if __name__ == "__main__":
    try:
        main()
    except (EnvironmentError, FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}")
