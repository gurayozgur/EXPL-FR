# Adapter weights

Place the released EXPL-FR adapter checkpoints in this folder:

```
adapter/
├── adapter_adaface_vitb_wf4m.pt
├── adapter_adaface_vits_wf4m.pt
├── adapter_adaface_rn100_wf4m.pt
└── adapter_adaface_rn100_ms1mv2.pt
```

The weights are available **[here](https://drive.google.com/drive/folders/1CHl0UPG7hZp-HzmBYm5VgQVz_hAb3Nbg?usp=sharing)**; to get access, please share your name, affiliation, and email in the request form.

Each checkpoint is a 4-layer MLP (~1M parameters, 512 -> 512) that maps the
frozen CLIP ViT-B/16 image space onto one specific frozen FR embedding space.
An adapter is only valid for the FR model it was trained against, listed in
`explfr/registry.py`. Download links are in the top-level README.
