# Zero-DCE — Noise Propagation Research

Research project investigating how perturbations propagate through Zero-DCE's eight
iterative curve transformations, and whether amplification can be controlled at
training time without changing the inference architecture.

**Read `notebooks/README.md` before proposing any code.**

---

## Working style

The user prefers a staged workflow:

- Explain the purpose in **simple, plain language** first.
- Give **one executable cell at a time**. Wait for the output. Interpret it before moving on.
- Don't launch long training sweeps or pick loss weights without evidence.
- Never assume a previous Colab runtime or its `/content` paths still exist.

---

## Repository layout

| Path | Contents |
|---|---|
| `Zero-DCE_code/` | Original Zero-DCE source. `Myloss.py` holds the earlier `L_noise` attempt. |
| `notebooks/01_zerodce_reproduction.ipynb` | Baseline reproduction and recorded metrics |
| `notebooks/02_phase_a_diagnostics.ipynb` | Phase A, Experiments 0–4 + validation |
| `notebooks/03_phase_b_experiments.ipynb` | Phase B, not yet run |
| `notebooks/99_archive_superseded.ipynb` | Abandoned work. Record only — do not reuse. |

**Branches.** Everything lives on **`master`** — the Phase A source changes
(`L_noise`, `--start_epoch`, the Colab compatibility fix), the notebooks and the docs.
All notebooks pull `master`. The old work branches `claude/nifty-ritchie-qpqakn` and
`claude/clever-gauss-thkwoj` were merged in and are now historical; do not develop on them.

`HANDOFF.md` is a dated record from the Phase A session (11 Nov) and still refers to the
old branch. Treat it as history — **this file is authoritative.**

---

## Model facts

- Checkpoint: `Epoch199.pth` on Drive, SHA256 `7c211c511a5213a898716cf65bee964e62ff04ffeb2991d20d7b5a28f7692154`
- 79,416 parameters · 85.54 GFLOPs @ 1200×900×3
- `model(x)` returns a 3-tuple: `(intermediate, enhanced, curves)`; curves are `[B, 24, H, W]` → `reshape(B, 8, 3, H, W)`
- Evaluation data: LOL eval15, 15 paired images. **Only `low/` is model input**; `high/` is the paired reference.
- SICE Part2 was unreachable, so LOL eval15 substitutes. Not comparable to the paper's numbers.

### Recurrence (code convention — use this one)

```
x_n = x_(n-1) + A_n * (x_(n-1)^2 - x_(n-1))
g_n = 1 + A_n * (2*x_(n-1) - 1)              local derivative
J_n = prod(g_i, i = 1..n)                     cumulative, clean trajectory
```

The paper writes `I + aI(1-I)`. Since `x^2 - x = -x(1-x)`, the code maps as `A = -a`.
Keep the two conventions distinct.

---

## Phase A results (complete)

| Measurement | Value |
|---|---|
| Recurrence reconstruction | MAE 0, max error 0 |
| Mean luminance | 0.05976 → 0.38788 |
| Frozen-curve variance amplification | 78.35× / 82.65× / 104.22× at σ = .005/.010/.020 |
| Mean cumulative `J²` at iteration 8 | 76.988 |
| p99 cumulative | 293.82 |
| Jacobian vs measured | 1.59% / 6.60% / 31.46% relative difference |
| Out-of-range output fraction | 5.02% / 7.09% / 10.66% |

Dark regions stay locally amplifying across all 8 iterations; brighter regions
increasingly attenuate. End-to-end variance is ~1.5% below frozen — **this is not
evidence of denoising.**

### Critical interpretation caveat

`J` is simultaneously the noise gain **and** the local contrast gain. 76.99× in
variance is 8.77× in standard deviation, which is also the factor by which dark-region
contrast is amplified. A pointwise curve multiplies detail and noise by the same
number, so it cannot change first-order SNR.

**Never report the amplification figure without this.** On its own it is not evidence
of quality degradation.

---

## Do not repeat

- **Do not build a regulariser that penalises `g²` or `J²`.** `g` is the contrast gain,
  so the model satisfies the penalty by dimming the image. Measured: PSNR 17.26 → 14.74
  at W50, → 9.21 at W400. That was not a mask bug; the objective pointed at the wrong quantity.
- **Do not reuse `W50` or any weight from the old sweeps.**
- **Do not use Sobel gradient magnitude as a flatness mask.** It is large for edges *and*
  noise. Use structure-tensor coherence.
- **Do not treat BRISQUE/NIQE as evidence.** They disagreed on this data (NIQE improved,
  BRISQUE worsened on 45/74).
- **Do not generalise mid-bright/bright findings.** Those regions are nearly empty in
  LOL eval15 — `bright` is 0 pixels in 9 of 15 images, max 1,180 of 240,000.

---

## Phase B hypothesis (current direction)

Decouple noise gain from contrast gain by leaving pointwise operation:

```
b = blur(x);  d = x - b
x_n = f_n(b + gamma*d) + (1 - gamma)*d
```

`gamma = 1` is bit-exact vanilla Zero-DCE; `gamma = 0` gives full brightening with
detail gain exactly 1. Later, `gamma` becomes a learned per-pixel map.

Figure of merit — the **decoupling ratio**:

```
R = amplification(structured perturbation) / amplification(unstructured perturbation)
```

Vanilla Zero-DCE, ZLEN, and any pointwise curve reformulation have `R = 1` identically.

### On ZLEN (Appl. Sci. 2024, 14, 2846)

Different method, same problem space — must be cited. ZLEN adds a noise-estimation
module, semantic attention and dropout (3.7 ms → 5.2 ms, +40%). Its "improved curve"
`LE + a*LE*(1 - LE + N)` is still **pointwise** in `LE`, so it cannot decouple anything;
whatever gain ZLEN achieves comes from its fusion module changing the predicted `a`,
not from the curve equation it presents as contribution #1.

### Constraints

Preserve the inference architecture. No denoising network. Aim for ≈ Zero-DCE cost
(blur ≈ 1% of FLOPs; measure with `thop`, and report wall-clock too — blurs are
memory-bound). Do not reduce noise by dimming or by smoothing. Evaluate enhancement
quality and perturbation robustness separately, with identical noise draws for paired
comparisons.

---

## Notebook hygiene (this bit the project once)

The original `ZERODCE_1.ipynb` used six globals that were **never defined in any saved
cell** — `device`, `BRIGHTNESS_BINS`, `SIGMAS`, `N_TRIALS`, `BASE_SEED`,
`LOW_DIR`-as-`Path`, `RESULTS_DIR`. They lived only in a runtime. A top-to-bottom run
always failed with `NameError` at the model-loading cell.

Rules that prevent a repeat:

- Every notebook starts with **one bootstrap cell that defines everything**.
- Experiment cells **read** globals; they never introduce them.
- Each notebook ends its bootstrap with a smoke test that must print `... OK`.
- Verify with `pyflakes` over the flattened notebook before committing.

`BASE_SEED = 42026` is **reconstructed, not recovered**. It only selects noise draws,
so Experiments 2 and 3 reproduce statistically, not bit-for-bit. If an
`experiment_*_metadata.json` on Drive records the real seed, substitute it.

---

## Results on Drive

`/content/drive/MyDrive/zero-dce/results/phase_a/experiment_{0,1,2,3,4}/` and
`.../phase_b/experiment_5/`. CSV and JSON outputs are authoritative over any value
read off a plot.
