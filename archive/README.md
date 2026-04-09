# Archive

This directory holds tracked artifacts that are no longer part of the active
runtime surface but are still worth preserving for provenance.

Current archive policy:
- `legacy-results/`: historical frontier-sweep and pilot outputs that were
  previously stored at the repo root. They are snapshots, not live inputs for
  the current reasoning v2, probe, or patching pipelines.

Active artifact areas stay outside this archive:
- `pressuretrace-frozen/`: frozen paper/probe fixtures used by the live
  reasoning-family workflows and tests.
- `results/`: ignored local run outputs.
- `data/manifests/` and `data/splits/`: live benchmark inputs and slices.
