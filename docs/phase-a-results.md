# Phase A — Complete Results

Every Phase A number in one place. Exact values from notebook output and exported CSVs.
Where the 11 November handoff recorded plot-read approximations, the exact figures are
used here.

**Setup:** baseline `Epoch199.pth` · LOL eval15, 15 paired images, `low/` as input only
· σ ∈ {0.005, 0.010, 0.020} · 30 noise draws per image per σ · no training.

> **Always report with context.** `J²` is simultaneously the noise gain and the local
> contrast gain. 76.99× in variance is 8.77× in standard deviation, which is also the
> dark-region contrast amplification. See `the-pointwise-limit.md`.

## Experiment 0 — baseline verification

| Measurement | Result |
|---|---|
| Parameters | 79,416 |
| Curve channels | 24 |
| Recurrence reconstruction | MAE 0, max 0 |
| Mean input luminance | 0.05976 |
| Mean enhanced luminance | 0.38788 |
| Curve parameter range | −0.474989 … −0.071696 |

Mean derivative gain by iteration:
`1.3157, 1.3064, 1.2930, 1.2795, 1.2584, 1.2347, 1.2028, 1.1626`

Fraction with gain > 1:
`99.96%, 99.96%, 99.93%, 99.64%, 99.54%, 98.80%, 97.25%, 87.43%`

## Experiment 1 — controlled scalar sweep

100,000 Monte-Carlo samples, seed 42, constant `A`.

By curve parameter:

| A | Empirical | First-order prediction | Error | Mean brightness shift |
|---|---|---|---|---|
| −0.10 | 1.9388× | 1.9366× | 0.029% | −0.000054 |
| −0.25 | 5.4069× | 5.3822× | 0.220% | −0.000248 |
| −0.40 | 9.3529× | 9.1806× | 0.839% | −0.000616 |

By signal level — **note the last two columns track each other**:

| Clean signal | Variance amp | Noise std amp | Contrast retention | Brightness shift |
|---|---|---|---|---|
| 0.05 | 19.3913× | 4.40× | 4.036 | −0.000908 |
| 0.10 | 6.7695× | 2.60× | 2.532 | −0.000491 |
| 0.25 | 1.2202× | 1.10× | 1.056 | −0.000099 |
| 0.50 | 0.31697× | 0.56× | 0.473 | −0.000020 |
| 0.75 | 0.13299× | 0.36× | 0.278 | −0.000013 |

## Experiment 2 — frozen vs end-to-end

| σ | Frozen 2A | End-to-end 2B | 2B reduction | Jacobian | 2A error |
|---|---|---|---|---|---|
| 0.005 | 78.3494× | 77.9792× | ~0.47% | 76.9911× | 1.59% |
| 0.010 | 82.6481× | 81.4364× | ~1.46% | 76.9884× | 6.60% |
| 0.020 | 104.2240× | 98.2316× | ~5.75% | 76.9810× | 31.47% |

Mean absolute curve-map change: 0.000324 / 0.000952 / 0.002897
Output out-of-range fraction: **5.02% / 7.09% / 10.66%**

The ~1.5% end-to-end reduction is **not** evidence of denoising.

## Experiment 3 — iteration-wise propagation

| Stage | σ=.005 | σ=.010 | σ=.020 |
|---|---|---|---|
| input | 1.000 | 1.000 | 1.000 |
| 1 | 1.737 | 1.737 | 1.737 |
| 2 | 3.014 | 3.014 | 3.016 |
| 3 | 5.207 | 5.210 | 5.223 |
| 4 | 8.991 | 9.006 | 9.068 |
| 5 | 15.419 | 15.485 | 15.753 |
| 6 | 26.409 | 26.675 | 27.784 |
| 7 | 45.246 | 46.301 | 50.943 |
| 8 | 78.349 | 82.648 | 104.224 |

Final-stage regional variance at σ=0.020 (input ≈ 0.0004 everywhere):

| Region | Input | Frozen | End-to-end | Jacobian predicted |
|---|---|---|---|---|
| dark | 0.000400 | 0.048251 | 0.045475 | 0.035653 |
| mid-dark | 0.000400 | 0.003015 | 0.002994 | 0.002823 |
| mid-bright | 0.000409 | 0.000495 | 0.000489 | 0.000476 |
| bright | 0.000415 | 0.000077 | 0.000076 | 0.000077 |

## Experiment 4 — per-iteration sensitivity (exact, from CSV)

| Iter | Mean local var gain | Mean cumulative | p95 cumulative | p99 cumulative | Frac \|g\|>1 |
|---|---|---|---|---|---|
| 1 | 1.73653 | 1.73653 | 1.97582 | 2.02164 | 0.99964 |
| 2 | 1.71292 | 3.01369 | 3.90273 | 4.09068 | 0.99955 |
| 3 | 1.67918 | 5.20620 | 7.66792 | 8.22880 | 0.99934 |
| 4 | 1.64603 | 8.98550 | 15.09841 | 16.61526 | 0.99642 |
| 5 | 1.59451 | 15.39704 | 29.64644 | 33.51197 | 0.99540 |
| 6 | 1.53813 | 26.32037 | 58.39435 | 68.19410 | 0.98796 |
| 7 | 1.46443 | 44.89917 | 115.65438 | 140.47749 | 0.97246 |
| 8 | **1.37530** | **76.98827** | **231.56501** | **293.81985** | 0.87429 |

Bootstrap CIs resample **images**, not pixels (15 images).

### Region coverage — a severe limitation

Pixel counts out of 240,000 per image:

| Region | Coverage |
|---|---|
| dark | 92,428 – 239,938; present in all 15 |
| mid-dark | 60 – 147,571; present in all 15 |
| mid-bright | **0 in 6 of 15**; often 1–80 pixels |
| bright | **0 in 9 of 15**; maximum 1,180 |

**Do not generalise mid-bright or bright findings from this dataset.**

## Final validation — Exp 3 vs Exp 4

| σ | n | Measured mean | Jacobian mean | Mean rel. diff | Median | SD |
|---|---|---|---|---|---|---|
| 0.005 | 15 | 78.34944 | 76.98827 | 1.59330% | 1.62074% | 0.36494% |
| 0.010 | 15 | 82.64809 | 76.98827 | 6.59936% | 6.81083% | 1.56776% |
| 0.020 | 15 | 104.22402 | 76.98827 | 31.45978% | 32.49462% | 8.22260% |

Growing disagreement is the second-order `A·ε²` term becoming visible.

## Baseline reproduction (context)

| Metric | Ours (T4) | Paper (2080Ti) |
|---|---|---|
| PSNR | 17.26 (LOL eval15) | 16.57 (SICE Part2) |
| SSIM | 0.556 | 0.59 |
| MAE | 32.91 | 98.78 |
| Params | 79,416 | 79,416 / 79,532 |
| FLOPs @1200×900×3 | 85.54 G | 84.99 G |
| Runtime (naive) | 114.04 ms | 2.5 ms |

Different test sets — not directly comparable. Runtime differs by measurement protocol
(CUDA-event timing, warm cuDNN), not hardware alone. The paper's own Tables 4 and 7
disagree on parameter count by 116.

## Superseded — record only

Early Laplacian/high-frequency proxy diagnostics:

| Analysis | Result |
|---|---|
| Flat-patch proxy | 34,172 patches; median amplification 2.15× |
| Proxy vs mean curve magnitude | r = 0.877 |
| Proxy vs darkness | r = 0.827 |
| Proxy by stage | 0.01332 (input) → 0.06367 (stage 8) |
| Error vs darkness | r = −0.224 |
| **Error vs input proxy** | **r = 0.065** — noise amplification does not drive error |
| NIQE / BRISQUE (74 images) | 4.267→3.931 / 20.343→25.946; BRISQUE worse on 45/74 |

`W_noise` training sweeps:

| Run | PSNR | SSIM | MAE |
|---|---|---|---|
| baseline | 17.26 | 0.556 | 32.91 |
| W50 original mask | 13.93 | 0.541 | 52.72 |
| W50 fixed mask | 14.74 | 0.555 | 47.91 |
| W400 | 9.21 | 0.325 | 86.57 |

The v1 sweep (0.1–2.0) did nothing: at W=2.0 the term was 0.67% of total loss. These
failed because the objective targeted the wrong quantity, not because of the mask.

## Raw outputs on Drive

`/content/drive/MyDrive/zero-dce/results/phase_a/experiment_{0,1,2,3,4}/`

CSV and JSON are authoritative over any value read off a plot.
