"""Which FR backbone each released adapter was trained against.

The adapter is FR-model specific: it maps the VLM image space onto one
particular FR embedding space, so an adapter must be paired with the FR
checkpoint named here. FR weights are not redistributed, see README.
"""

from dataclasses import dataclass
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapter"


@dataclass(frozen=True)
class Target:
    key: str
    label: str
    adapter: str          # file name inside adapter/
    fr_config: str        # config path relative to CVLFACE_RUN_V1
    fr_source: str        # where to obtain the FR checkpoint


REGISTRY = {
    "vitb_wf4m": Target(
        "vitb_wf4m", "AdaFace ViT-B / WebFace4M",
        "adapter_adaface_vitb_wf4m.pt",
        "models/vit/configs/v1_base.yaml",
        "CVLface: minchul/cvlface_adaface_vit_base_webface4m",
    ),
    "vits_wf4m": Target(
        "vits_wf4m", "AdaFace ViT-S / WebFace4M",
        "adapter_adaface_vits_wf4m.pt",
        "models/vit/configs/v1_small.yaml",
        "CVLface: minchul/cvlface_adaface_vit_small_webface4m",
    ),
    "rn100_wf4m": Target(
        "rn100_wf4m", "AdaFace R100 / WebFace4M",
        "adapter_adaface_rn100_wf4m.pt",
        "models/iresnet/configs/v1_ir101.yaml",
        "CVLface: minchul/cvlface_adaface_ir101_webface4m",
    ),
    "rn100_ms1mv2": Target(
        "rn100_ms1mv2", "AdaFace R100 / MS1MV2",
        "adapter_adaface_rn100_ms1mv2.pt",
        "models/iresnet/configs/v1_ir101.yaml",
        "CVLface: minchul/cvlface_adaface_ir101_ms1mv2",
    ),
}

DEFAULT_TARGET = "vitb_wf4m"


def get_target(key: str) -> Target:
    if key not in REGISTRY:
        raise KeyError(f"unknown target '{key}'. Available: {', '.join(REGISTRY)}")
    return REGISTRY[key]
