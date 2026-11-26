# Related Work

## ZLEN — the one that must be beaten

*A Zero-Reference Low-Light Image-Enhancement Approach Based on Noise Estimation.*
Cao, Niu, Zhu, Li. **Appl. Sci. 2024, 14, 2846.** DOI 10.3390/app14072846

Same problem space as ours: Zero-DCE amplifies noise, fix it without references.

### What they do

1. **Modified curve** — `LE_n = LE_(n-1) + a_n·LE_(n-1)·(1 - LE_(n-1) + N(x))`
2. **Noise estimation** — colour map `C(x) = x / mean_c(x)`, then
   `N(x) = max(|∇x C|, |∇y C|)`
3. **Semantic-aware attention** fusing the noise map with image features
4. **Dropout** in the last two layers of the curve-estimation network
5. **Noise loss** — `L_nol = (1/S)·Σ(SNR − W)²`, W a constant matrix

Trained on SICE Part1 (2,416 images, 512×512), Adam, lr 1e-4, batch 8.

### Reported results on LOL

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Zero-DCE | 14.861 | 0.562 | 0.335 |
| **ZLEN** | **19.457** | **0.786** | **0.235** |
| LLFlow-SKF (full-reference) | 26.798 | 0.879 | 0.105 |

Runtime: Zero-DCE 0.0037 s → ZLEN 0.0052 s, **+40%**.

### Our critique

**Their contribution #1 cannot do what they claim.** `N(x)` is a fixed precomputed map,
so with `a` and `N` fixed at a pixel the curve is **still pointwise in `LE`**. Its slope
is `g = 1 + a(1 − 2·LE + N)`. The `N` term shifts the gain, but detail and noise at that
pixel are still multiplied by that same shifted `g`. See `the-pointwise-limit.md`.

Whatever ZLEN gains must come from the fusion/attention module changing *which* `a` is
predicted — a spatial effect — not from the curve equation.

**But their target was better than ours.** `L_nol` pushes **SNR**, a ratio, toward a
constant. A ratio can be improved. Our old `g²` penalty targeted noise alone and dragged
signal with it.

### Open discrepancy — resolve before publishing

ZLEN reports Zero-DCE on LOL at PSNR **14.861** / SSIM **0.562**.
Our reproduction on LOL eval15: PSNR **17.26** / SSIM **0.556**.

SSIM agrees; PSNR differs by 2.4 dB. Likely causes: different checkpoint (ours is our
own 200-epoch training, not the released `Epoch99`), different resizing, different
preprocessing. This changes who appears to win and **must be settled**.

## Still to check — do this before investing months

Base/detail decomposition with separate gain control is textbook image processing. A
reviewer will raise it. Specifically review:

**Tone mapping / decomposition**
- Unsharp masking (classical)
- Durand & Dorsey, *Fast Bilateral Filtering for the Display of HDR Images*, 2002
- Paris et al., *Local Laplacian Filters*, 2011

**Retinex-decomposition LLIE** — different split (reflectance/illumination) but
reviewers will connect it
- RetinexNet (BMVC 2018), KinD (ACM MM 2019), KinD++ (IJCV 2021), URetinex-Net (CVPR 2022)

**Self-supervised / zero-reference LLIE**
- Zero-DCE++ , RUAS (CVPR 2021), SCI (CVPR 2022), PairLIE (CVPR 2023)

**Noise-aware LLIE**
- Anything pairing Zero-DCE with multi-scale or frequency decomposition

The defence, if the decomposition itself is deemed old: the contribution is the
impossibility result, the corrected account of ZLEN, the iterative zero-reference
setting, and doing it with no ground truth.
