# Session Handoff — 26 November 2026
## Phase A reinterpreted · Phase B designed · Repository rebuilt

**Status:** Phase A complete and reinterpreted. Phase B designed but **not yet run**.
**Next action:** run the first cell of `notebooks/03_phase_b_experiments.ipynb`.
**Branch:** everything is on `master`.

This document is written from scratch. You should be able to read only this and
continue, without the earlier conversation.

---

# 1. What Zero-DCE is

Zero-DCE brightens dark photographs. What makes it unusual is that it never sees a
"correct" bright version of any image during training — no paired examples, no ground
truth. That is what "zero-reference" means.

It works like this. A small network (79,416 parameters) looks at a dark image and
predicts a set of **curve parameters** — one number per pixel, per colour channel, per
iteration. Those numbers describe a simple arithmetic rule for making a pixel brighter.
That rule is then applied **eight times in a row**.

The rule, exactly as the released code implements it:

```
x_n = x_(n-1) + A_n * (x_(n-1)^2 - x_(n-1))
```

`x` is a pixel value between 0 and 1. `A_n` is the predicted parameter for iteration
`n`. Run it eight times and a dark pixel becomes a bright one.

**A note that has caused confusion before.** The paper writes the curve as
`I + a*I*(1 - I)`. The code writes `x + A*(x^2 - x)`. Since `x^2 - x = -x(1 - x)`,
these are the same thing with `A = -a`. The paper's `a` is positive; the code's `A` is
negative. Everything in this project uses the **code** convention. Keep them distinct
or signs will flip under you.

### The two derived quantities everything rests on

If you hold the curve parameters fixed and ask "if this pixel changed slightly, how much
does the output change?", you get the **local derivative**:

```
g_n = 1 + A_n * (2*x_(n-1) - 1)
```

Chain all eight together and you get the **cumulative Jacobian**:

```
J_n = g_1 * g_2 * ... * g_n
```

`g > 1` means that step magnifies small changes. `g < 1` means it shrinks them.
These two quantities are the backbone of the entire project.

---

# 2. The aim

Low-light photos are noisy. Zero-DCE brightens them. The widespread intuition —
stated in the Zero-DCE paper's own limitations, and asserted by follow-up papers like
ZLEN — is that brightening **amplifies that noise**, and that this is a defect worth
fixing.

Our aim was:

> Understand exactly how perturbations propagate through the eight curve iterations,
> work out which part of it is caused by the curve transformation itself, and find out
> whether harmful amplification can be reduced **at training time**, without changing
> the inference architecture, without adding a denoising network, and ideally without
> adding any inference-time cost.

The constraint on inference cost is deliberate. Zero-DCE's selling point is that it is
tiny and fast. A fix that makes it slow is not interesting — that solution already
exists, and it is called "use a denoiser".

---

# 3. The hypothesis, and how it changed

### What we started with

> Repeated curve transformations amplify existing noise, especially in dark or flat
> regions, and this amplification compounds across the eight iterations. A lightweight
> training-side regulariser should be able to reduce it.

### What Phase A measured

The amplification is real and large. 76.99× in variance, cumulative over eight
iterations. Dark regions amplify throughout; bright regions attenuate.

### What this session worked out

**The original hypothesis is close to a tautology, and the proposed fix cannot work.**

This is the single most important result of the session, and §6 explains it in full.
The short version: the number that describes noise amplification is *the same number*
that describes contrast amplification. They are not two quantities that happen to be
correlated — they are literally one quantity. So "the curves amplify noise" is a
restatement of "the curves brighten dark detail", which is the method's purpose.

### What we are testing now

> Because a pointwise curve multiplies detail and noise by the same factor, it cannot
> change signal-to-noise ratio at first order. Decoupling the two therefore requires
> abandoning pointwise operation. Splitting the image into a smooth **base** layer and
> a **detail** layer, brightening the base with the existing curve, and giving the
> detail its own independent gain should separate brightness control from noise control
> — at roughly Zero-DCE's cost, with vanilla Zero-DCE recoverable as an exact special
> case.

---

# 4. The repository

Everything lives on **`master`**. The old work branches
(`claude/nifty-ritchie-qpqakn`, `claude/clever-gauss-thkwoj`) were merged in on
26 November and are historical. Do not develop on them.

```
CLAUDE.md                    Project context, read automatically by Claude each session
docs/
  README.md                  Index of documentation
  phase-a-results.md         Every Phase A number in one place
  the-pointwise-limit.md     The core theoretical result
  related-work.md            ZLEN and what still needs checking
  handoffs/
    2026-11-11-phase-a.md    Phase A session record (historical)
    2026-11-26-phase-b-design.md   This document
notebooks/
  README.md
  01_zerodce_reproduction.ipynb
  02_phase_a_diagnostics.ipynb
  03_phase_b_experiments.ipynb
  99_archive_superseded.ipynb
Zero-DCE_code/               Original source; Myloss.py holds the old L_noise attempt
```

## What each notebook does

### `01_zerodce_reproduction.ipynb` — 18 cells
Rebuilds the baseline from nothing. Clones the repo, mounts Drive, downloads the SICE
training data, trains for 200 epochs producing `Epoch199.pth`, downloads LOL eval15,
then measures full-reference metrics, parameter count, FLOPs, runtime, and renders the
curve-parameter maps.

Run this only if you need to rebuild the checkpoint. It takes hours.

### `02_phase_a_diagnostics.ipynb` — 27 cells
The diagnostic study. One `BOOTSTRAP` cell followed by Experiments 0–4 and a final
cross-check. Nothing here trains anything — it all runs against the frozen baseline
checkpoint.

### `03_phase_b_experiments.ipynb` — 12 cells
**Not yet run.** A session-start cell plus Experiments 5.0 through 5.3. Experiment 5.0
is a hard gate; 5.1–5.3 are meaningless until it passes.

### `99_archive_superseded.ipynb` — 22 cells
Abandoned work, kept as a record. The Laplacian proxy diagnostics and all three
`W_noise` training sweeps. Several cells will not run — they reference runtime state
that no longer exists. **Do not reuse any result or weight from this notebook.**

## How the notebooks are structured, and why

Every notebook opens with **one bootstrap cell that defines everything** and ends that
cell with a smoke test printing `BOOTSTRAP OK` or `PHASE B READY`. Experiment cells only
*read* globals; they never create them.

This rule exists because of a specific failure. The original `ZERODCE_1.ipynb` used
seven globals that were never defined in any saved cell — `device`, `BRIGHTNESS_BINS`,
`SIGMAS`, `N_TRIALS`, `BASE_SEED`, `LOW_DIR`-as-a-`Path`, and `RESULTS_DIR`. They
existed only in the memory of a live Colab session. Cell 34 defined `DEVICE`
(uppercase); cell 36 read `device` (lowercase). A top-to-bottom run **always** failed
with `NameError` at the model-loading cell, on every runtime, deterministically. It only
ever appeared to work because the original session had those names defined in cells that
were later edited or deleted.

Before committing a notebook, flatten its code cells into one file and run `pyflakes`
over it. The only undefined name that should appear is `display`, which Colab injects.

---

# 5. Phase A — what we did and what came out

Phase A was purely diagnostic. No training, no loss changes, no new checkpoint. The
method was to inject synthetic Gaussian noise of *known* standard deviation and measure
exactly what came out.

Setup throughout: baseline `Epoch199.pth`, LOL eval15 (15 paired images, **only the
`low/` folder is model input**), noise levels σ ∈ {0.005, 0.010, 0.020}, 30 independent
noise draws per image per level.

## Experiment 0 — does our maths match the code?

Fed all 15 images through the model, then reconstructed the eight-step recurrence by
hand from the returned curve maps and compared.

| Measurement | Result |
|---|---|
| Parameters | 79,416 |
| Curve channels | 24 (8 iterations × 3 channels) |
| Reconstruction error | **MAE 0, max 0** |
| Mean luminance | 0.05976 → 0.38788 |
| Mean gain per iteration | 1.3157, 1.3064, 1.2930, 1.2795, 1.2584, 1.2347, 1.2028, 1.1626 |
| Fraction with gain > 1 | 99.96% → 87.43% |
| Curve parameter range | −0.474989 to −0.071696 |

Exact reconstruction means every later analytical result is about the real
implementation, not an idealisation of it. This is the foundation everything else
stands on.

## Experiment 1 — isolate the recurrence from the network

Pure Monte-Carlo on scalars: 100,000 samples, constant `A`, no network involved. The
point was to see the curve's behaviour with nothing else mixed in.

| Clean signal | Variance amplification | Noise std amplification | Contrast retention |
|---|---|---|---|
| 0.05 | 19.3913× | **4.40×** | **4.04×** |
| 0.10 | 6.7695× | **2.60×** | **2.53×** |
| 0.25 | 1.2202× | **1.10×** | **1.06×** |
| 0.50 | 0.31697× | **0.56×** | **0.47×** |
| 0.75 | 0.13299× | **0.36×** | **0.28×** |

The headline reading at the time was "strong brightness dependence — dark amplifies,
bright attenuates". Correct, but look at the last two columns. Noise gain and contrast
gain track each other almost exactly. **The answer to the whole project was sitting in
this table and we did not see it.** See §6.

First-order prediction error was under 1% here, so the maths is sound.

## Experiment 2 — is it the curve, or the network?

Two branches, same noise draws:
- **2A frozen** — predict curves from the *clean* image, hold them fixed, apply to noisy input
- **2B end-to-end** — push the noisy image through the whole network, let curves be re-predicted

| σ | Frozen (2A) | End-to-end (2B) | Jacobian prediction | 2A prediction error |
|---|---|---|---|---|
| 0.005 | 78.3494× | 77.9792× | 76.9911× | 1.59% |
| 0.010 | 82.6481× | 81.4364× | 76.9884× | 6.60% |
| 0.020 | 104.2240× | 98.2316× | 76.9810× | 31.47% |

The two branches are close, so the fixed curve transformation accounts for nearly all
the response. End-to-end is ~1.5% lower — **this is not evidence that the network
denoises.** It is one number under one measurement.

Mean absolute curve-map change under noise: 0.000324 / 0.000952 / 0.002897. Small.

Output out-of-range fraction: **5.02% / 7.09% / 10.66%**. Raw outputs were kept
unclipped; clipping was analysed separately. Remember this number — it matters in §8.

## Experiment 3 — where in the chain does it happen?

Measured variance after *every* stage, not just the end.

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

Growth is progressive through all eight iterations, not concentrated at the start. Each
step acts on perturbations the previous steps already magnified, which is why it
compounds.

Final-stage variance by region at σ=0.020 (input variance ≈ 0.0004 everywhere):

| Region | Frozen output variance |
|---|---|
| dark | 0.048251 |
| mid-dark | 0.003015 |
| mid-bright | 0.000495 |
| bright | 0.000077 |

## Experiment 4 — attribute it per iteration and per pixel

Computed `g_n` and `J_n` at every pixel and channel along the clean trajectory, grouped
by original-image luminance. Bootstrap confidence intervals resample **images**, not
pixels — pixels within an image are spatially correlated and treating them as
independent would produce fake-tight intervals.

| Iteration | Mean local variance gain | Mean cumulative | p95 cumulative | p99 cumulative |
|---|---|---|---|---|
| 1 | 1.73653 | 1.73653 | 1.97582 | 2.02164 |
| 2 | 1.71292 | 3.01369 | 3.90273 | 4.09068 |
| 3 | 1.67918 | 5.20620 | 7.66792 | 8.22880 |
| 4 | 1.64603 | 8.98550 | 15.09841 | 16.61526 |
| 5 | 1.59451 | 15.39704 | 29.64644 | 33.51197 |
| 6 | 1.53813 | 26.32037 | 58.39435 | 68.19410 |
| 7 | 1.46443 | 44.89917 | 115.65438 | 140.47749 |
| 8 | **1.37530** | **76.98827** | **231.56501** | **293.81985** |

By iteration 8 the fraction of pixels with |g| > 1 is ~96% in dark regions, ~43%
mid-dark, ~6% mid-bright, near zero in bright.

### A severe dataset limitation

Region pixel counts, out of 240,000 per image:

| Region | Coverage |
|---|---|
| dark | 92,428 – 239,938 — present in all 15 |
| mid-dark | 60 – 147,571 — present in all 15 |
| mid-bright | **0 in 6 of 15 images**; often 1–80 pixels |
| bright | **0 in 9 of 15 images**; maximum 1,180 |

The claim "bright regions attenuate" rests on a few hundred pixels in a handful of
images. It is not supported by this dataset. One reviewer comment would kill it.

## Final validation — Experiment 3 against Experiment 4

| σ | Measured mean | Jacobian mean | Mean rel. diff | Median | SD |
|---|---|---|---|---|---|
| 0.005 | 78.34944 | 76.98827 | 1.59330% | 1.62074% | 0.36494% |
| 0.010 | 82.64809 | 76.98827 | 6.59936% | 6.81083% | 1.56776% |
| 0.020 | 104.22402 | 76.98827 | 31.45978% | 32.49462% | 8.22260% |

The first-order Jacobian explains small perturbations well and degrades as they grow.
That degradation is not mysterious: the recurrence has a second-order term `A*ε²` the
derivative ignores, and it becomes visible as ε grows.

## Phase A conclusions, stated honestly

1. The recurrence reconstructs exactly.
2. Curve response is strongly brightness dependent.
3. Variance grows progressively through all eight iterations.
4. Frozen and end-to-end are close; the small end-to-end reduction proves nothing.
5. The first-order Jacobian is accurate for small perturbations only.
6. A minority of pixels carry far higher cumulative sensitivity than the mean (p99 = 293.82 vs mean 76.99).

**Limits:** synthetic additive Gaussian noise added on top of images that are *already*
noisy; 15 images; mid-bright and bright effectively absent; the frozen Jacobian
excludes the network's curve-prediction pathway entirely.

---

# 6. The reinterpretation — the most important section

Phase A measured everything correctly. The interpretation was wrong.

## The argument, in plain words

The Zero-DCE curve is **pointwise**. One pixel goes in, one pixel comes out. It never
looks at neighbours.

Take two adjacent pixels in a dark area:

- real detail: `0.050` and `0.055` → the difference is `0.005`
- noise sitting on top: about `±0.004`

Apply a curve whose slope here is `g = 8.8`:

- the detail `0.005` becomes `0.005 × 8.8 = 0.044`
- the noise `0.004` becomes `0.004 × 8.8 = 0.035`

**Both were multiplied by the same number.** The ratio between detail and noise — the
thing your eye actually judges — did not move.

## Why it is unavoidable

The slope `g` *is* the contrast gain. There are not two separate numbers, one for signal
and one for noise. For any smooth pointwise function:

```
noise amplification = (contrast amplification)²
```

So `J² = 76.99` and "this curve multiplies dark-region contrast by 8.77×" are **the same
statement**. Phase A measured, very carefully, that Zero-DCE brightens dark regions.

**Never quote the 76.99× figure without the 8.77× contrast figure beside it.** Quoted
alone it implies a defect that the number does not demonstrate.

## Three independent confirmations already in our own data

1. **Experiment 1's table.** Noise std gain 4.40 vs contrast gain 4.04 at signal 0.05;
   2.60 vs 2.53 at 0.10; 1.10 vs 1.06 at 0.25. They track.

2. **Diagnostic Test 3, from months ago.** `Pearson r(error, input noise, flat) = 0.065`,
   recorded at the time with the note *"noise amplification does NOT meaningfully drive
   full-reference error."* That was correct and should have stopped us then.

3. **The failed training runs.** See below.

## Why the earlier `L_noise` work failed — and it was not a bug

| Run | PSNR | SSIM | MAE |
|---|---|---|---|
| baseline (W=0) | 17.26 | 0.556 | 32.91 |
| W50, original mask | 13.93 | 0.541 | 52.72 |
| W50, fixed mask | 14.74 | 0.555 | 47.91 |
| W400 | 9.21 | 0.325 | 86.57 |

The old objective penalised `relu(g - 1)` on a flatness mask. `g` is the contrast gain.
We asked the model to reduce contrast gain, and it did — by darkening the image. PSNR
collapsed. **The model obeyed exactly.**

The first sweep (W 0.1–2.0) did nothing because at W=2.0 the term was 0.67% of total
loss. The second (W 20–400) was large enough to bite, and dimmed the output.

This was blamed on a buggy mask. The mask *was* flawed — it used **Sobel gradient
magnitude**, which is large for edges *and* for noise and cannot separate them. But
fixing it only moved PSNR from 13.93 to 14.74, still far below the 17.26 baseline. The
objective was pointed at the wrong quantity. No mask and no weight can repair that.

**Consequence: do not build a regulariser that penalises `g²` or `J²`.** It will fail
the same way, more slowly.

---

# 7. Where ZLEN sits

ZLEN (*A Zero-Reference Low-Light Image-Enhancement Approach Based on Noise Estimation*,
Appl. Sci. 2024, 14, 2846) attacks the same problem. We are **not** duplicating it, but
it must be cited and compared.

| | ZLEN | Us |
|---|---|---|
| Changes the curve equation | Yes — adds a noise term | No |
| Adds inference modules | Yes — noise estimation, semantic attention, fusion, dropout | No |
| Inference cost | 3.7 ms → **5.2 ms (+40%)** | target ≈ unchanged |
| Needs noise maps at test time | Yes | No |
| Noise loss | `L_nol = mean(SNR − W)²`, heuristic | derived from propagation analysis |
| Analysis of *how* noise propagates | **None** | our entire Phase A |

## Their contribution #1 does not do what they claim

ZLEN's "improved higher-order curve" is:

```
LE_n = LE_(n-1) + a_n * LE_(n-1) * (1 - LE_(n-1) + N(x))
```

`N(x)` is a fixed map precomputed from the input's colour-map gradients. With `a` and
`N` fixed at a pixel, this is **still a pointwise function of `LE`**. Its slope is
`g = 1 + a*(1 - 2*LE + N)`. The `N` term shifts the gain — but detail and noise at that
pixel are still multiplied by that same shifted `g`.

**So ZLEN's curve modification cannot decouple anything either.** Whatever improvement
ZLEN achieves must come from its fusion/attention module changing *which* `a` gets
predicted — a spatial, non-pointwise effect — not from the equation they present as
their first contribution.

This is a genuine finding and is ours to publish.

## But they aimed better than we did

Their loss pushes **SNR** — a *ratio* — toward a constant. A ratio is a quantity that
can actually be improved. Our old `g²` penalty targeted noise alone and dragged signal
down with it. Their derivation is loose and their implementation heavy, but their target
quantity was more correct. Worth acknowledging.

## An open discrepancy to resolve

ZLEN's Table 1 reports Zero-DCE on LOL at **PSNR 14.861, SSIM 0.562**. Our reproduction
on LOL eval15 measures **PSNR 17.26, SSIM 0.556**. SSIM agrees closely; PSNR differs by
2.4 dB. Possible causes: different checkpoint (ours is our own 200-epoch training, not
the released `Epoch99`), different resizing, or different preprocessing. **This must be
resolved before any published comparison**, because it changes who appears to win.

---

# 8. What harm is actually left?

If first-order amplification is a non-problem, what *can* genuinely go wrong? Three
candidates.

### A — Clipping. Real, already measured.
5.02% / 7.09% / 10.66% of output values fall outside [0, 1]. When a noisy pixel is
pushed past 1.0 and clamped, information is destroyed permanently. Clipping is **not**
pointwise-reversible, so the SNR argument gives it no protection. Clipped noise also
becomes biased and blotchy. Fixable *locally* — restrain the curve only where saturation
threatens, which does not dim the rest of the image.

### B — The second-order term. Real, but tiny.
The `A*ε²` term the derivative ignores shows up as a small brightness bias. Experiment 1
measured shifts of about −0.0006. Too small to build on. It is, however, the complete
explanation for the 1.59% → 6.60% → 31.47% growth in prediction error.

### C — The curve estimator wobbling. Unmeasured, most interesting.
The network *predicts* `A` from the image, and unlike the curve it has a **receptive
field** — it sees neighbourhoods. So noise at one pixel can shift the curve across a
whole patch, producing **blotches and colour shifts**: errors that are spatially
*structured*.

Experiment 2 concluded this pathway was small because end-to-end variance was only ~1.5%
below frozen. **But pixelwise variance is blind to spatial structure.** A smooth blotch
across 50×50 pixels can have low pixelwise variance and still be the ugliest artefact in
the picture. Our measurement cannot see the thing that matters most.

---

# 9. Phase B — the plan

## The idea

Detail and noise can only be told apart by one fact: **real detail is spatially
correlated, noise is not.** Using that requires looking at more than one pixel — which
means leaving pointwise operation. That is the whole move.

Before each curve step, split the image:

- **base** `b = blur(x)` — smooth, large structures, overall brightness
- **detail** `d = x - b` — fine texture, edges, **and noise**

Then brighten the base with the curve, and give the detail its own gain.

## The modified recurrence

```
b = blur(x_(n-1))
d = x_(n-1) - b

x_n = f_n(b + gamma*d) + (1 - gamma)*d
```

`f_n` is the **unmodified** Zero-DCE curve. Expanding it:

```
x_n = f_n(b) + [gamma*g_b + (1 - gamma)] * d + gamma^2 * A_n * d^2
```

| gamma | detail gain | meaning |
|---|---|---|
| **1** | `g_b` | **bit-exact vanilla Zero-DCE** |
| **0** | 1 | full brightening, noise never amplified at all |
| between | between | partial |

Brightness is steered by `A` acting on the base. Noise is steered by `gamma`. Two knobs
where there was one.

**Vanilla Zero-DCE is the exact special case where the two gains are forced equal.** The
method is a strict generalisation, not a replacement — which is an excellent position
for a paper, and means training can be initialised at a known-good point.

## The figure of merit

```
R = amplification(structured perturbation) / amplification(unstructured perturbation)
```

For vanilla Zero-DCE, for ZLEN, and for **any** pointwise curve: `R = 1` identically,
always. Any `R > 1` is something no pointwise method can achieve. That is the
contribution, and it is measurable.

Across eight iterations the ratio compounds as `prod(g) / prod(s)` — so the iterative
structure that currently compounds noise would instead compound **selectivity**.

## The experiments, in order

### 5.0 — exactness gate. **Hard gate.**
Implement the decomposition, set `gamma = 1`, confirm it reproduces vanilla Zero-DCE.
Expect `max_abs_error` ≈ `1e-7` (float32 round-off). **Above `1e-5`, stop** — the
decomposition is wrong and nothing downstream means anything. Also reports how much
image energy lives in the detail layer per brightness region.

### 5.1 — gamma sweep, no noise. Does brightness survive?
Sweep `gamma ∈ {1.0, 0.7, 0.5, 0.3, 0.0}` and measure mean luminance, clipping fraction,
SSIM vs vanilla, and PSNR/SSIM vs the paired reference.

- **Want:** `mean_luminance` roughly **flat** across gamma. That is the core claim.
- **If it tracks gamma:** the base is leaking brightness — raise `BASE_BLUR_SIGMA` and retry.
- **Expect** `ssim_vs_vanilla` to fall as gamma drops. That is the honest trade: a
  *global* gamma suppresses real texture along with noise. How fast it falls tells you
  how much headroom a learned per-pixel gamma has.

### 5.2 — noise amplification vs gamma
Re-run the Experiment 3 protocol under the modified recurrence.

- **Consistency check:** at `gamma = 1` this must recover 78.35 / 82.65 / 104.22.
  Statistically, not bit-for-bit — see the seed note in §11.
- At `gamma = 0`, detail gain is exactly 1, so amplification should land near **1×** at
  unchanged brightness.

### 5.3 — the decoupling ratio. **The money figure.**
Two perturbations of identical variance: white noise (lives in the detail layer) and a
smooth low-frequency perturbation (lives in the base). Measure the response to each.

| Result | Reading |
|---|---|
| `R ≈ 1.0` at gamma = 1 | control arm correct — pointwise sits at exactly 1 |
| `R > 1` at gamma < 1 | decoupling is real. **This is the paper.** |
| `R ≈ 1.0` at *every* gamma | the decomposition is not separating structure from noise — switch to a guided filter before concluding anything |

## Cost

Blur per iteration ≈ 1% of 85.5 GFLOPs. Extra output channels add low single digits.
Call it **"about Zero-DCE cost"**, not zero.

Measure it with `thop` rather than estimating, and report **wall-clock too** — blurs are
memory-bandwidth-bound, so timing looks worse than FLOPs suggest. The comparison that
matters is against ZLEN's +40%.

---

# 10. Risks that could kill this

**The model will cheat by smoothing.** The old objective let it dim the image; this one
lets it blur. Driving `gamma → 0` everywhere satisfies "less noise" and returns a
plastic, texture-free picture. The loss **must** contain a structure-preservation term.

**Halos.** Brightening a blurred base while suppressing detail is the classic halo
recipe. A plain Gaussian will produce them. Use an edge-aware base — a guided filter is
O(N) via box filters and keeps edges in the base where they belong.

**The crux: telling texture from noise.** `gamma` must be near 1 on real texture and near
0 on flat noisy regions. Use **structure-tensor coherence** — the ratio of the local
structure tensor's eigenvalues. An edge has one dominant direction; noise has none. Same
gradient magnitude, completely different coherence. **Do not use gradient magnitude.**
That was the flaw in the old mask.

**Novelty risk — take this seriously.** Base/detail decomposition with separate gain
control is textbook image processing: unsharp masking, Durand & Dorsey's bilateral tone
mapping (2002), local Laplacian filters. A reviewer who knows that literature will say
*"this is Durand–Dorsey with a CNN"*. The defensible claims are:

1. The impossibility result — pointwise curve LLIE has `R = 1` identically.
2. The corrected mechanism account of ZLEN.
3. Decomposition inside an *iterative, zero-reference* loop with per-iteration gamma.
4. Doing it with **no ground truth**; tone mapping had references and human tuning.
5. The measurement framework.

Claims 1 and 2 are the strong ones. **Check the literature before investing months** —
specifically the Retinex-decomposition line (RetinexNet, KinD, URetinex-Net) and
self-supervised LLIE (SCI, RUAS, PairLIE).

**Dataset.** 15 images is too few, and mid-bright/bright are effectively absent. Add
SICE, LSRW, or at minimum the full LOL training split before publishing anything
regional.

**Noise realism.** We add white Gaussian noise to images that are already noisy. Real
low-light noise is signal-dependent (Poisson-Gaussian) and spatially correlated after
demosaicing. We measured sensitivity to an artificial probe, not real-world behaviour.

---

# 11. Things that will trip you up

**`BASE_SEED = 42026` is reconstructed, not recovered.** The original was never saved
anywhere. It only selects which noise draws occur, so Experiments 2 and 3 reproduce
*statistically*, not bit-for-bit. If an `experiment_*_metadata.json` on Drive records
the real seed, substitute it:

```
/content/drive/MyDrive/zero-dce/results/phase_a/experiment_3/experiment_3_metadata.json
```

**Only `eval15/low` goes into the model.** `eval15/high` is the paired reference. An
earlier draft accidentally fed `high` as input. Preserve the distinction.

**Keep `model_module` (the module) distinct from `model`/`net` (the network).** The
original notebook rebound the name `model` from one to the other, which is a trap.

**Do not trust BRISQUE or NIQE here.** On our data NIQE improved (4.267 → 3.931) while
BRISQUE worsened (20.343 → 25.946, worse on 45 of 74 images). They disagree, so neither
is evidence.

**CSV and JSON outputs on Drive beat any value read off a plot.** Several numbers in the
11 November handoff were marked "plot-read, approximate". The exact Experiment 4 values
are now in `docs/phase-a-results.md`.

---

# 12. Exactly what to do next

1. Open a **fresh Colab** notebook. Runtime → Change runtime type → **GPU**.
2. Upload `notebooks/03_phase_b_experiments.ipynb`.
3. Run the **SESSION START** cell. Confirm it prints **`PHASE B READY`**.
4. Run **Experiment 5.0**. Confirm **`GATE PASSED`** and `max_abs_error` ≈ `1e-7`.
   - If it fails, stop. Report the error. Do not continue.
5. Run **5.1**. Check whether `mean_luminance` stays flat across gamma.
6. Run **5.2**. Check that `gamma = 1` recovers ~78 / ~83 / ~104.
7. Run **5.3**. This is the decisive one.

Then, depending on 5.3:

- **`R > 1` at gamma < 1** → the mechanism works. Next: make gamma a learned per-pixel
  map driven by structure-tensor coherence, design the zero-reference loss with a
  structure-preservation term, and do the literature check.
- **`R ≈ 1` everywhere** → the Gaussian decomposition is not separating structure from
  noise. Try a guided filter. If that also gives `R ≈ 1`, the premise is dead — pivot to
  clipping (Candidate A in §8), or write up the negative result, which is publishable
  and genuinely useful.

---

# 13. Housekeeping done this session

**Diagnosed a deterministic notebook failure.** `device` was never assigned in any saved
cell — the only occurrences were `device=device` keyword arguments. Cell 36 failed every
time, on every runtime. Not flaky.

**Split `ZERODCE_1.ipynb` (54 cells) into four notebooks**, added bootstrap cells
defining all seven previously-missing globals, collapsed repeated `git pull` cells into
one per notebook, and dropped cells 7, 13, 20, 21, 29, 53 (duplicate pulls, one fully
commented-out cell, one empty cell).

**Verified with `pyflakes`** over the flattened notebooks: no undefined names except
`display`. This also caught a bug in the generated files themselves — newlines had been
stripped from authored cells, which would have made them open in Colab as a single line.

**Merged everything into `master`** and repointed all notebooks from
`claude/nifty-ritchie-qpqakn` to `master`.

**Wrote `CLAUDE.md`** so future sessions load the context automatically.

---

# 14. Glossary

| Term | Meaning |
|---|---|
| **pointwise** | one pixel in, one pixel out, no neighbours consulted |
| **`A_n`** | curve parameter at iteration `n` — code convention, negative |
| **`a`** | the paper's curve parameter, positive; `A = -a` |
| **`g_n`** | local derivative at iteration `n`; also the local contrast gain |
| **`J_n`** | cumulative product of `g` through iteration `n` |
| **frozen curves (2A)** | curves predicted from the clean image, held fixed |
| **end-to-end (2B)** | noisy image through the whole network, curves re-predicted |
| **base / detail** | smooth layer and high-frequency remainder, `d = x - blur(x)` |
| **gamma** | Phase B detail-gain knob; 1 = vanilla, 0 = no amplification |
| **decoupling ratio `R`** | structured response ÷ unstructured response; 1 for all pointwise methods |
| **structure coherence** | eigenvalue ratio of the local structure tensor; separates edges from noise |
