# Baselines

The learners the paper compares against, kept here so the whole comparison
lives in one clone.

| Directory | What it is | Licence | What is ours |
|---|---|---|---|
| `dreamerv3/` | github.com/danijar/dreamerv3 @ `e3f0224`, vendored | MIT, Danijar Hafner | three environment bridges; upstream code untouched |
| `dreamerv3-torch/` | github.com/NM512/dreamerv3-torch @ `6ef8646`, vendored | MIT, NM512 | two environment bridges, three campaign scripts, a 94-line patch |
| `openvla/` | our code; `openvla/openvla-7b` is loaded from Hugging Face | MIT, OpenVLA (the weights) | all of it |

The two vendored directories keep the upstream `LICENSE` and upstream
`README.md` exactly as published. What we added is recorded in `PROVENANCE.md`
next to them, and our edits to upstream files are kept as a patch in
`OUR_CHANGES.diff` so the modification is readable without diffing against a
clone you do not have.

Both DreamerV3 families are used in the paper — the JAX one is the vanilla
online arm, the PyTorch one the demonstration-seeded arm — which is why both
are here rather than one.

`openvla/` is not a vendored repository: OpenVLA-7B is loaded through
`transformers` from the Hugging Face hub, so there is no upstream source tree to
carry. Only our fine-tuning and evaluation code is here.

Nothing here ships model weights. The two DreamerV3 sources are about 9 MB
together and `openvla/` is a few tens of kilobytes; the OpenVLA-7B base weights
(≈ 14 GB) are downloaded on first use, and the LoRA adapters our runs produced
are not shipped, being useless without the base.
