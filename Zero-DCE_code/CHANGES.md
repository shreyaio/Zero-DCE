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

- **Sign convention: code vs. paper's Eq. 1.** The paper's prose gives the
  LE-curve as `LE(x) = x + α·x·(1-x)`. The released `model.py` actually computes
  `x + r·(x²-x)`, which is the algebraic *negative* of that (`x²-x = -x(1-x)`).
  Found while implementing `L_noise` below (a sanity check against a synthetic
  dark pixel showed positive `r` *darkens* rather than brightens). This doesn't
  affect reproduction fidelity — the released network was trained end-to-end
  against its own formula and learns the correspondingly negative `r` values in
  dark regions — but it matters for any new code (like `L_noise`) that needs to
  replicate the curve's actual math rather than the paper's prose description.

## 2026-10-27 — noise-gain regularizer (contribution phase, opt-in via `--W_noise`)

Added a training-time loss term to penalize the curve iterations for amplifying
noise in flat/low-texture regions, motivated by diagnostic tests run on the
Tier-1 reproduction model (see the "Zero-DCE Failure-Mode Diagnostics" doc):
noise amplification in flat patches correlated strongly with curve magnitude
(Pearson r=0.877) and accelerated across the 8 iterations.

- **`Myloss.py`**: new `L_noise` class. Uses the curve's own analytic local
  gain `g_n(x) = 1 + A_n·(2x_{n-1}-1)` (the derivative of `model.py`'s actual
  recurrence, bounded in `[0,2]`, consistent with the sign-convention note
  above) as the noise-amplification estimate at each iteration, by standard
  first-order noise-propagation through a pointwise nonlinearity. Penalizes
  `ReLU(g_n - 1)²` (only amplification, not attenuation), weighted by (a) an
  iteration index `n/8` (increasing — matches the measured accelerating noise
  growth) and (b) a flatness mask computed once from the original input image
  (Sobel gradient magnitude, `exp(-20·|∇|)`, detached from the graph) so the
  penalty targets flat regions and leaves `L_spa`'s edge-contrast objective
  alone. Recomputes the 8-iteration recurrence internally from `x` and `r`
  rather than modifying `model.py`'s return signature, to keep `model.py` and
  `lowlight_test.py` completely untouched.
- **`lowlight_train.py`**: added `--W_noise` (float, **default `0.0`**) and
  `loss_noise = config.W_noise*L_noise(img_lowlight, A)`, added into the total
  loss. Default `0.0` means the baseline Tier-1 reproduction config is
  bit-for-bit unchanged unless `--W_noise` is explicitly passed for a
  contribution run.
- Verified locally (CPU, synthetic tensors, no GPU needed) before pushing:
  loss is differentiable w.r.t. the curve parameters; a flat near-black region
  under a strongly brightening curve is penalized far more than a textured
  region under the same curve; a curve that darkens (gain<1) is correctly
  unpenalized; penalty scales with curve magnitude. One bug caught and fixed
  during this check: an early version of `flat_weight` normalized gradient
  magnitude relative to each image's own max gradient, which inverted the
  intended behavior on perfectly uniform synthetic inputs (the epsilon-floor
  gradient got normalized up to ~1 instead of ~0); replaced with an absolute,
  non-per-image-normalized scale.
- **Follow-up fix (same day, caught on first real Colab run):** `L_noise`'s
  Sobel kernels were built with plain `torch.FloatTensor(...)`, leaving them on
  CPU while every other tensor in training is on CUDA — `RuntimeError: Input
  type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be
  the same`. `L_spa`'s existing kernels in this same file hardcode `.cuda()`
  into the tensor construction for the same reason (the training loop never
  calls `.cuda()` on the loss modules themselves, only on tensors) — matched
  that established pattern rather than introducing a different one. This was
  missed by the earlier local CPU-only sanity check (no GPU in that
  environment), which is why it only surfaced on the first real Colab run.
