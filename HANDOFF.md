# Zero-DCE Reproduction + Noise-Gain Regularizer — Project Handoff

Last updated: 2026-11-07. Written for a fresh session (Claude or human) to pick this up
with full context. Read this top to bottom before touching code or Colab.

## What this project is

Undergrad research paper for an image/video processing class. Two phases:

1. **Phase 1 (done):** reproduce the original Zero-DCE paper (Guo et al., CVPR2020 /
   TPAMI2021 extended version, "Zero-Reference Deep Curve Estimation for Low-Light
   Image Enhancement") using the official PyTorch repo, on Google Colab free-tier T4.
2. **Phase 2 (in progress):** a lightweight training-side contribution — a **noise-gain
   regularizer** added to Zero-DCE's loss function, motivated by diagnostics showing
   the curve amplifies sensor noise in flat/dark regions. Goal: fix this with *zero
   added inference-time parameters* (just a new loss term), unlike every prior
   noise-fix in this literature, which bolts on a second network.

Repo: `https://github.com/shreyaio/Zero-DCE`, branch `claude/nifty-ritchie-qpqakn`
(NOT merged to `master` — always `git checkout`/`pull` this branch explicitly).

Two Claude Docs track the project narrative in more detail than this file:
- [Zero-DCE Reproduction Audit](https://claude.ai/artifact/7HkrHbi3PFBuwBTQj7FK6h) — per-table/figure reproducibility scoring from the paper
- [Zero-DCE Failure-Mode Diagnostics](https://claude.ai/artifact/DpQ3svrgvp38RPgfLSMVux) — diagnostic tests, literature review, regularizer design/results

## Environment facts that matter

- Colab free-tier T4. **Sessions disconnect without warning** and wipe `/content` —
  the repo clone, any pip installs, and the extracted LOL dataset all need
  re-fetching after a disconnect. Checkpoints/results on Drive survive.
- Google Drive mounted at `/content/drive/MyDrive/zero-dce/` — this is the only
  persistent storage. See the file map below for what's already there.
- Always use the idempotent clone/pull pattern (clones if `/content/Zero-DCE`
  is missing, otherwise fetches+checks out the branch) — every cell in this
  project assumes you're on the latest commit of `claude/nifty-ritchie-qpqakn`.
- A Colab kernel that already has `Myloss`/`lowlight_train` imported from
  earlier in the session will **not** pick up a `git pull`'s changes without
  `importlib.reload(Myloss); importlib.reload(lowlight_train)` — this has
  bitten us twice, always reload explicitly after pulling.
- My own sandbox (this Claude session, not Colab) has network access locked to
  a narrow allowlist — `github.com` works, but `arxiv.org`, `huggingface.co`,
  `mdpi.com`, `sciencedirect.com`, `pmc.ncbi.nlm.nih.gov`, `proceedings.iclr.cc`,
  `goo.gl`, `gitee.com`, `alphaxiv.org` all fail to resolve. Paper PDFs had to
  be uploaded by the user directly; don't assume a fresh session can WebFetch
  academic sites either without checking.

## Repo changes so far (see `Zero-DCE_code/CHANGES.md` for full detail on each)

Commits on `claude/nifty-ritchie-qpqakn`, in order:

| Commit | What |
| --- | --- |
| `0103b10` | PyTorch/Pillow compat: `Image.ANTIALIAS`→`Image.LANCZOS`, `clip_grad_norm`→`clip_grad_norm_` |
| `017f129` | Logged: official training zip has 2,002 images, not the paper's stated 2,422 |
| `d66df9a` | Added `--start_epoch` CLI arg + loop fix, for resume-from-checkpoint across disconnects |
| `cfdf89c` | Added `L_noise` regularizer to `Myloss.py`, wired via `--W_noise` (default `0.0` — baseline unaffected unless explicitly enabled) |
| `693cfe7` | Fixed device-mismatch bug: `L_noise`'s Sobel kernels weren't on `.cuda()` |
| `1e23bd1` | Fixed `L_noise`'s flatness mask: was computed on raw dark pixels (too broad, see below) |

Also two **known, deliberately-not-fixed discrepancies** between the paper and the
released code (logged in `CHANGES.md`, kept as-is per reproduction-fidelity policy):
- Loss weights in `lowlight_train.py` (`TV×200`, `col×5`, `exp×10`) don't match
  the paper's stated `W_tv=20`, `W_col=0.5` — code's actual weights were kept.
- The curve recurrence in `model.py` (`x + r·(x²−x)`) is the algebraic *negative*
  of the paper's stated `x + α·x·(1−x)`. Found while deriving `L_noise`'s gain
  formula — doesn't affect the released model (it was trained end-to-end against
  its own formula) but matters for any new code touching the curve's math.

## Phase 1 — Reproduction results (Tier 1 scope only; Tiers 2/3 not attempted)

Trained 200 epochs, repo defaults, on the official 2,002-image training set, T4,
~2.5-3 hrs wall clock (paper claims ~30 min on a 2080Ti — expected hardware gap).
Final checkpoint: `/content/drive/MyDrive/zero-dce/checkpoints/Epoch199.pth`.

**Blocker hit and resolved:** the paper's actual Table 6 test set (SICE Part2, 767
paired images) is unreachable — the official `csjcai/SICE` repo's Google Drive link
for Part2 is a dead `goo.gl` shortlink, and the only remaining host is Baidu
Netdisk (impractical from Colab, no account, heavily throttled for non-China IPs).
**Substituted the LOL dataset** (`eval15` split, 15 paired images, via
`huggingface.co/datasets/geekyrakshit/LoL-Dataset`) per explicit user decision.
**Numbers below are NOT comparable to the paper's published Table 6** — different
test set entirely. Use them only as this reproduction's own baseline for later
A/B comparison against the regularized model.

| Check | Result |
| --- | --- |
| Params | **79,416** — exact match to paper's Table 4 (Table 7 says 79,532 — pre-existing paper-internal inconsistency) |
| FLOPs (1200×900×3) | **85.54 G** (thop) vs paper's 84.99 G — ~0.6% off, effectively exact |
| Runtime (1200×900×3) | **114.04 ms** (T4, naive `time.time()`) vs paper's 2.5 ms (2080Ti) — not comparable as measured, likely a CUDA-event-timing/warm-cudnn difference, not raw hardware alone |
| LOL eval15 (n=15) | **PSNR 17.26, SSIM 0.556, MAE 32.91** |
| Sanity check | Pretrained `Epoch99.pth` on DICM/LIME: clear brightening, detail recovery, no color shift — qualitatively matches paper's Fig 1/14 |
| Curve maps (Fig 3/6 style) | Reproduced: high curve parameter in dark regions, low in bright — matches paper's described pattern |

## Phase 2 — Diagnostics (5 tests, run on the Tier-1 baseline model, 74 DICM+LIME images + 15 LOL eval15 images)

Working hypothesis tested: Zero-DCE's curve amplifies pre-existing sensor noise in
flat/dark regions, roughly proportional to how hard the curve pushes, and this
compounds across the 8 iterations.

| Test | Result | Reading |
| --- | --- | --- |
| 1. Amplification vs. curve `\|A\|` / darkness | r=0.877 (vs `\|A\|`), r=0.827 (vs darkness), both p≈0, n=34,172 flat patches. Median amplification 2.15x | **Confirmed, strongly** |
| 2. Noise growth across 8 iterations | Stage 0→8: 0.0133→0.0637. Stage-over-stage growth 0.0023→0.0123 (iter 1→8) | **Accelerating, not linear** — later iterations do disproportionately more damage |
| 3. Stratified full-reference error on LOL eval15 | r(error, darkness)=−0.224 (negative!), r(error, input noise, flat only)=0.065 (n=4,170, practically negligible) | **Noise amplification does NOT drive PSNR/SSIM/MAE** — full-reference metrics are largely blind to this failure mode |
| 4. NIQE/BRISQUE before/after (n=74) | NIQE 4.27→3.93 (improves), BRISQUE 20.3→25.9 (worsens, 45/74 images worse) | Dissociation confirms structural/noise-specific damage, not just naturalness |
| 5. Zoomed worst-case crops | Worst 3 files by amplification: 14.jpg, 18.jpg, 26.jpg. Same 3 files independently show the worst BRISQUE deltas (+51, +33, +64) | Two independent measurement methods agree on the same failure cases |

**Net conclusion:** noise amplification is real, severe, and mechanistically tied to
curve magnitude and iteration depth — but it's a **perceptual/structural quality
problem, not an accuracy-metric problem**. The regularizer's case must rest on
BRISQUE/visual evidence, explicitly not on expected PSNR/SSIM gains.

## Literature review (full PDFs read, not just search snippets)

| Work | Mechanism | Verdict |
| --- | --- | --- |
| Self-DACE (arXiv 2308.08197) | 4 loss terms in curve stage, **none noise-related**; separate 18-22 layer Denoising-Net handles all noise | Confirms the gap cleanly. Full pipeline w/ denoiser = 9x params/FLOPs (0.699M vs 0.079M) for only +0.65 PSNR — corroborates our Test 3 finding |
| "Breaking Down the Darkness" (arXiv 2111.15557) | 3 separate U-Nets (IAN/ANSN/CAN) | Heavy, but gives a **formal proof**: `I = R∘L + N = (R+N̄)∘L`, `N̄=N/L` — dividing out small illumination algebraically amplifies effective noise. Cite as theoretical backing alongside our empirical Test 1 |
| ZLEN (MDPI *Appl. Sci.* 2024, 14(7):2846) | Same total-loss structure as Zero-DCE + 1 term (`L_nol`), but routes noise map through a learned fusion/attention module first | **Closest prior art.** Still adds real parameters (fusion module), noise signal is an external gradient heuristic + SNR target, not iteration-weighted |
| CLODE (ICLR 2025, poster only — full paper unreachable) | Continuous Neural-ODE curve + separate Noise2Noise-based noise-removal module | Same "separate module" pattern. Motivation framed as "convergence," not noise-compounding — related but distinct from our framing |

**Precise novelty claim:** no work found does a *zero-architecture-change,
analytically-derived-from-the-curve's-own-math, iteration-weighted* fix. (Not the
broader "no loss-only fix exists" — ZLEN is close enough that this narrower framing
is the honest one.)

## The regularizer: design, bugs found, and current state

**Design.** The curve recurrence `x_n = x_{n-1} + A_n·(x_{n-1}²−x_{n-1})` has an
exact analytic local gain `g_n(x) = 1 + A_n·(2x_{n-1}−1)`, bounded in `[0,2]` by
construction. By first-order noise propagation through a pointwise nonlinearity,
this gain *is* the noise-amplification factor. Current `Myloss.py`:

```python
class L_noise(nn.Module):

    def __init__(self):
        super(L_noise, self).__init__()
        kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def flat_weight(self, x):
        gray = torch.mean(x, dim=1, keepdim=True)
        lifted = torch.clamp(gray, min=0) ** 0.4          # gamma-lift, see "bug #3" below
        gx = F.conv2d(lifted, self.weight_x, padding=1)
        gy = F.conv2d(lifted, self.weight_y, padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
        return torch.exp(-6.0 * grad_mag)

    def forward(self, x, r):
        r_iters = torch.split(r, 3, dim=1)
        n_iters = len(r_iters)
        fw = self.flat_weight(x).detach()
        x_prev = x
        loss = 0.0
        for n, A_n in enumerate(r_iters, start=1):
            gain = 1 + A_n * (2 * x_prev - 1)
            excess = F.relu(gain - 1.0)
            w_n = n / n_iters                              # weights later iterations more
            loss = loss + w_n * torch.mean((excess ** 2) * fw)
            x_prev = x_prev + A_n * (torch.pow(x_prev, 2) - x_prev)
        return loss / n_iters
```

Wired into `lowlight_train.py`: `loss_noise = config.W_noise*L_noise(img_lowlight, A)`,
added to the total loss, `--W_noise` defaults to `0.0`.

**Bug #1 (caught locally, before any Colab run):** an earlier `flat_weight`
normalized gradient magnitude relative to each image's own max gradient
(`grad_mag / (amax(grad_mag) + eps)`) — on a perfectly uniform synthetic input this
inverts the intended behavior (the epsilon-floor gradient normalizes up to ~1
instead of ~0). Fixed to an absolute, non-per-image scale.

**Bug #2 (caught on first real Colab run):** kernels built with plain
`torch.FloatTensor(...)`, staying on CPU while training runs on CUDA →
`RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
(torch.FloatTensor) should be the same`. Fixed by hardcoding `.cuda()` into the
kernel construction, matching the existing `L_spa` kernels' pattern in this same
file (the loss modules themselves are never `.cuda()`'d, only individual tensors).

**Bug #3 (caught via a `W_noise` sweep + visual check, the big one):** a sweep
from 20 to 400 showed the amplification/BRISQUE metrics improving monotonically
with weight exactly as designed, but LOL PSNR/SSIM/MAE collapsing in lockstep
(PSNR 17.26→9.21, MAE 32.91→86.57 at W=400). A visual side-by-side revealed mean
image brightness dropping substantially and *uniformly across the whole frame* at
every weight — the network was satisfying the penalty by brightening less
everywhere, not by selectively protecting noisy flat patches. Root cause,
confirmed by measuring `flat_weight` directly on real bundled test images: on raw
dark pixels, a genuinely dark photo has near-zero local contrast almost
everywhere — including regions that would reveal real structure once brightened —
so the mask was flagging **61.5%-81.7% of pixels** as "protect from penalty,"
far too broad. Fixed with the gamma-lift (`gray**0.4`) shown above, which reveals
real structure hidden by darkness before judging flatness. An alternative
(per-image min-max stretch) was tried and rejected first — numerically unstable
on near-constant patches, where a tiny max-min denominator blows up residual
sensor noise into looking like real structure (verified via a synthetic test,
worth re-deriving if curious: `flat_dark` scored *lower* than `dark_structured`
under min-max stretch, the exact wrong ordering).

Verified the fix differentiates by actual content, not just "is it dark": on
`01.jpg`, protected fraction drops 61.5%→30.9%; on `18.jpg`, 81.7%→21.2%; but on
`1.bmp` (genuine large sky regions), it correctly **rises** 60.3%→74.1%.

## Loss-scale sensitivity (important, re-derive before trusting any new sweep)

A raw-magnitude check on a real batch, baseline model:

| Term | Raw value | Actual weight | Weighted value |
| --- | --- | --- | --- |
| TV | 0.001049 | 200 | 0.209749 |
| spa | 0.011930 | 1 | 0.011930 |
| col | 0.002315 | 5 | 0.011573 |
| exp | 0.038886 | 10 | 0.388859 |
| **noise (raw)** | **0.002108** | varies | — |

Sum of the 4 existing weighted terms ≈ 0.622. At `W_noise=2.0` (sweep v1's max),
the weighted noise term was only **0.67% of total loss** — explains why sweep v1
(0.1-2.0) showed zero effect. Needed `W_noise≈100-200` to reach 25-40% share.
**If `L_noise`'s formula changes again, this ratio should be re-checked** — the
raw magnitude is sensitive to the exact penalty shape.

## Sweep results (all fine-tunes: 15 epochs from the Tier-1 baseline checkpoint, not from-scratch — this was a deliberate curriculum choice, see note below)

**Sweep v1 (W_noise 0.1-2.0)** — `/content/drive/MyDrive/zero-dce/noise_sweep/sweep_results.json`. No meaningful movement on any metric (see loss-scale note above — range was ~100x too small).

**Sweep v2 (W_noise 20/50/100/200/400), buggy mask** — `/content/drive/MyDrive/zero-dce/noise_sweep_v2/sweep_results.json`:

| label | med_amp | r(amp,a) | NIQE in→out | BRISQUE in→out | LOL PSNR | LOL SSIM | LOL MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 2.15 | 0.877 | 4.27→3.93 | 20.3→25.9 | 17.26 | 0.556 | 32.91 |
| W20 | 2.11 | 0.820 | 4.27→4.01 | 20.3→23.1 | 16.18 | 0.569 | 40.65 |
| W50 | 1.98 | 0.695 | 4.27→3.72 | 20.3→19.7 | 13.93 | 0.541 | 52.72 |
| W100 | 1.85 | 0.560 | 4.27→3.76 | 20.3→17.5 | 12.33 | 0.495 | 63.28 |
| W200 | 1.37 | 0.395 | 4.27→3.88 | 20.3→16.9 | 10.29 | 0.401 | 77.80 |
| W400 | 1.18 | 0.301 | 4.27→3.96 | 20.3→17.4 | 9.21 | 0.325 | 86.57 |

Mechanism confirmed working (amp ratio, correlation, BRISQUE all move the right
way monotonically) but PSNR/SSIM/MAE collapse — this is the sweep that triggered
the visual check → Bug #3 diagnosis above.

**W50 re-run with the fixed mask** — `/content/drive/MyDrive/zero-dce/noise_sweep_v3_fixed/`:

| label | med_amp | r(amp,a) | NIQE in→out | BRISQUE in→out | LOL PSNR | LOL SSIM | LOL MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 2.15 | 0.877 | 4.27→3.93 | 20.3→25.9 | 17.26 | 0.556 | 32.91 |
| old W50 (buggy mask) | 1.98 | 0.695 | 4.27→3.72 | 20.3→19.7 | 13.93 | 0.541 | 52.72 |
| **new W50 (fixed mask)** | **2.02** | **0.738** | 4.27→3.73 | **20.3→20.6** | **14.74** | **0.555** | **47.91** |

SSIM essentially fully recovered (0.555 vs baseline 0.556). PSNR/MAE partially
recovered (not fully). BRISQUE improvement is smaller than the buggy version but
still meaningful: baseline clearly *damages* BRISQUE (20.3→25.9), fixed-mask W50
is roughly *neutral* (20.3→20.6) — no longer worse than the input.

## Where this session left off — READ THIS FIRST

A combined cell (numbers + visual side-by-side of baseline vs. new-fixed-W50 on
`18.jpg`/`01.jpg`/`1.bmp`) was handed to the user to run, **but the output was
never pasted back before this handoff was written.** The numbers above (SSIM
0.555, PSNR 14.74, etc.) are already confirmed from the prior numeric-only run —
what's *not yet confirmed* is whether the fixed mask actually solved the uniform-
dimming problem visually (mean brightness per panel), only that SSIM recovering
strongly suggests it did. **First thing next session: get that image + confirm
brightness numbers no longer show the steep, uniform decline seen in the buggy-mask
sweep.**

## Open decisions / next steps

1. Confirm the W50-fixed visual check (above).
2. Decide: is W50 with the fixed mask good enough, or re-sweep a few points
   (e.g. 50/80/120/150) now that the mask behaves correctly and may support a
   higher weight without the old collapse? User's last framing leaned toward "fix
   it, don't re-sweep extensively" but this wasn't explicitly closed off.
3. Once a weight is chosen: train the **actual contribution model** — 200 epochs,
   **from scratch** (not a 15-epoch fine-tune — that was a cheap sweep-ranking
   proxy, not meant to be the reported model), with that `W_noise`.
   - Note on curriculum: all sweeps so far fine-tuned from the already-converged
     baseline rather than training from scratch with all 5 losses from a random
     init. This was a deliberate choice (avoids the risk of the network finding
     "don't brighten at all" as a trivial early local optimum) — worth keeping
     this framing in mind when training the final model: consider whether a
     gradual `W_noise` ramp over the 200 epochs would be safer than a constant
     weight from epoch 0, especially given how sensitive the collapse behavior
     turned out to be.
4. Re-run diagnostic Tests 1, 2, 4, 5 (and ideally the LOL eval set) on the final
   contribution model, to produce the actual before/after comparison for the
   paper.
5. Write up: reproduction section (Phase 1 results + honest caveats), method
   section (the `L_noise` derivation + both bugs as methodology narrative — this
   is genuinely good material, a real research iteration story), results section
   (lead with BRISQUE/amplification, be explicit that PSNR/SSIM are not expected
   to improve and explain why via Test 3), related work (the 4-paper comparison
   table above), limitations.

## Drive file map (everything persisted so far)

```
/content/drive/MyDrive/zero-dce/
├── checkpoints/                      Tier-1 baseline, Epoch0-199.pth (199 = final)
├── data/train_data.zip               cached official training zip (2,002 images)
├── results/
│   ├── sanity_check/                 pretrained-weights sanity check (grid + crops)
│   ├── lol_eval15/                   baseline enhanced LOL images + metrics.json
│   ├── curve_maps/                   Fig 3/6-style curve parameter map visualization
│   └── diagnostics/
│       ├── patch_records.json        Test 1/2 raw per-patch data (74 DICM+LIME images)
│       ├── test1_noise_vs_alpha.png
│       ├── test2_noise_per_iteration.png
│       ├── test3_records.json        Test 3 raw per-patch data (15 LOL eval15 images)
│       ├── test3_error_vs_darkness_noise.png
│       ├── test4_iqa_records.json    Test 4 NIQE/BRISQUE per-image data
│       ├── test5_worst_crops.png
│       └── *_worst_crop_*.png
├── noise_sweep/                      sweep v1 (W_noise 0.1-2.0), sweep_results.json
├── noise_sweep_v2/                   sweep v2 (W_noise 20-400, buggy mask), sweep_results.json
└── noise_sweep_v3_fixed/
    └── W50/                          fixed-mask W50 checkpoints (Epoch0-14.pth)
    comparison_results.json           baseline vs old-W50 vs new-W50 numeric comparison
```

`/content/lol_dataset/` (LOL eval15 raw images) and `/content/Zero-DCE/` (the repo
clone) are **not persisted** — both get re-fetched by the idempotent setup cells
each session (see any earlier cell in this project for the pattern: check
`os.path.isdir`, clone/download if missing, otherwise just `cd`/mount).
