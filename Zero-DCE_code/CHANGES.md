# CHANGES.md

Log of every modification made to the original Zero-DCE (CVPR2020) code for the
Colab reproduction. Minimal-change policy: only what's required to run on current
PyTorch/Colab, nothing stylistic. Each entry: file, line, what changed, why.

## 2026-10-26 — Colab/PyTorch compatibility fixes

- **`dataloader.py`, line 50**: `Image.ANTIALIAS` → `Image.LANCZOS`.
  `Image.ANTIALIAS` was deprecated in Pillow 9.1 and removed entirely in Pillow 10,
  which current Colab installs by default. `Image.LANCZOS` is the same resampling
  filter under its non-deprecated name and is a no-op change in behavior.

- **`lowlight_train.py`, line 78**: `torch.nn.utils.clip_grad_norm(...)` →
  `torch.nn.utils.clip_grad_norm_(...)`. The non-underscore `clip_grad_norm` was
  removed from PyTorch years ago in favor of the in-place `clip_grad_norm_`, which
  is the direct drop-in replacement (same gradient-clipping behavior).

## Known discrepancies NOT changed (flagged, not "fixed")

- **Loss weights** in `lowlight_train.py` (`Loss_TV = 200*L_TV(A)`, `loss_col =
  5*mean(...)`, `loss_exp = 10*mean(...)`) do not match the weights stated in the
  paper's §5.1 / Eq. 8 (`W_col=0.5`, `W_tv_A=20`, implicit weight 1 on L_spa and
  L_exp). Per reproduction-fidelity policy, the code's actual (released, default)
  weights are kept as-is, since they are what produced the authors' released
  checkpoint and reported numbers — not the paper's prose description.
- `F.tanh` in `model.py` is deprecated but was left untouched since it still runs;
  will revisit only if it errors on the Colab PyTorch version actually encountered.

## 2026-10-26 — resume-from-checkpoint support

- **`lowlight_train.py`, argparse block**: added `--start_epoch` (default `0`).
- **`lowlight_train.py`, training loop**: `for epoch in range(config.num_epochs):`
  → `for epoch in range(config.start_epoch, config.num_epochs):`.
  Why: `--load_pretrain`/`--pretrain_dir` already existed as CLI args to resume
  *weights* from a checkpoint, but the loop always counted epochs from 0
  regardless, so resuming after a Colab disconnect would silently train past
  the intended 200 total epochs. This two-line change lets the orchestrating
  notebook pass the correct `start_epoch` (detected from the highest
  `Epoch{N}.pth` already saved to Drive) so total epochs trained across any
  number of resumptions still sums to `num_epochs`. No change to the loss
  functions, optimizer, or default hyperparameters.

## Data notes

- The training-data zip linked from `README.md` (Google Drive id
  `1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3`) contains **2,002** flat `.jpg` images in
  its root, not the 2,422 the paper's §5.1 text describes for the Part1 training
  split. This is the authors' own officially released training package, used
  as-is (no re-derivation from raw SICE Part1), so the reported discrepancy is
  between the paper's prose and the authors' own released data, not something
  introduced by this reproduction.
