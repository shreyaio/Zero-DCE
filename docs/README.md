# Documentation

## Start here

**New to the project, or returning after a break?** Read the most recent handoff in
`handoffs/`. It is written from scratch and is self-contained.

## Contents

| Document | What it is | Read when |
|---|---|---|
| `handoffs/` | Dated session records | Resuming work |
| `phase-a-results.md` | Every Phase A number, exact | You need a figure to quote |
| `the-pointwise-limit.md` | The core theoretical result | Before designing any regulariser |
| `related-work.md` | ZLEN analysis + literature still to check | Writing up, or checking novelty |

`../CLAUDE.md` holds the condensed version and is loaded automatically by Claude each
session. These documents are the long form.

## The one-paragraph version

Zero-DCE brightens dark images by applying a pointwise curve eight times. We set out to
show those iterations amplify noise, and measured it carefully: 76.99× cumulative
variance amplification, concentrated in dark regions. We then worked out that this
number is *also* the contrast amplification — a pointwise curve multiplies detail and
noise by the same factor, so it cannot change signal-to-noise ratio at first order. That
makes the original finding near-tautological and makes gain-penalty regularisation
structurally impossible, which explains why earlier training attempts dimmed the image
rather than cleaning it. Phase B therefore leaves pointwise operation: split the image
into base and detail, brighten the base with the existing curve, and give the detail its
own gain. Vanilla Zero-DCE is the exact special case where the two gains are forced
equal.

## Conventions

- Code convention `x + A(x² − x)` throughout; the paper's `a` maps as `A = −a`
- CSV and JSON on Drive beat any value read off a plot
- Amplification figures are never quoted without their contrast-gain counterpart
