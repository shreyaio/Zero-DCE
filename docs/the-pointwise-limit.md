# The Pointwise Limit

> A pointwise curve cannot change signal-to-noise ratio at first order.
> This is the central theoretical result of the project.

## Statement

Let `f` be any smooth function applied independently to each pixel — no neighbours
consulted. Let `g(x) = f'(x)` be its slope at value `x`.

For a pixel at value `s` carrying detail `Δ` and noise `ε`, both small:

```
f(s + Δ) - f(s) ≈ g(s)·Δ        detail is multiplied by g
f(s + ε) - f(s) ≈ g(s)·ε        noise  is multiplied by g
```

Both scale by the **same** factor. Therefore

```
contrast-to-noise ratio  Δ/ε  is unchanged by f,
```

and

```
noise variance amplification = (contrast amplification)²
```

`g` is not "the noise gain" and separately "the contrast gain". It is one number that
is both.

## Consequences

**1. "Iterative curves amplify noise" is near-tautological.** Amplifying dark-region
detail is what the method is *for*. Measuring the amplification measures the
enhancement.

**2. Our 76.99× is also 8.77×.** `sqrt(76.99) = 8.77` is the factor by which
dark-region contrast is amplified. Report both together, always.

**3. Gain-penalty regularisation cannot work.** Penalising `g²` or `J²` reduces contrast
gain, so the model complies by darkening the output. Observed: PSNR 17.26 → 14.74 at
W50, → 9.21 at W400. Not a tuning failure — a structural one.

**4. No curve reformulation escapes it.** Quadratic, cubic, higher-order, logarithmic,
square-root, Anscombe-transformed, or ZLEN's noise-augmented curve — all pointwise, all
locked.

**5. Decoupling requires leaving pointwise operation.** The only information that
distinguishes signal from noise is that real detail is spatially correlated and noise is
not. Using it requires looking at more than one pixel.

## The figure of merit this implies

```
R = amplification(structured perturbation) / amplification(unstructured perturbation)
```

Every pointwise method has `R = 1` identically. `R > 1` is achievable only by a
non-pointwise operation, which makes `R` a clean, measurable statement of contribution.

## Where this was already visible in our data

| Clean signal | Noise std gain | Contrast gain |
|---|---|---|
| 0.05 | 4.40× | 4.04× |
| 0.10 | 2.60× | 2.53× |
| 0.25 | 1.10× | 1.06× |

From Experiment 1, recorded before the implication was noticed.

Also: Diagnostic Test 3 found `Pearson r(error, input noise, flat) = 0.065` and was
annotated at the time *"noise amplification does NOT meaningfully drive full-reference
error."*

## What harm remains

Three candidates, since first-order amplification is not one:

| | Harm | Status |
|---|---|---|
| **A** | **Clipping** — 5–10% of outputs leave [0,1]; clamping destroys information irreversibly and is not pointwise-reversible | real, measured, unexploited |
| **B** | **Second-order term** `A·ε²` — brightness bias ≈ −0.0006 | real, too small to matter |
| **C** | **Curve-estimator instability** — the network has a receptive field, so noise shifts `A` across whole patches, giving *spatially structured* blotches | unmeasured; pixelwise variance is blind to it |

Phase B pursues the decomposition route. Candidate A is the fallback if it fails.
