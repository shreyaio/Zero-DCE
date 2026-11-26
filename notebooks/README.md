# Notebooks

Run order, and what each is for. Upload directly to Colab; enable GPU via
*Runtime → Change runtime type → GPU*.

| Notebook | Purpose | Status |
|---|---|---|
| `01_zerodce_reproduction.ipynb` | Reproduce baseline Zero-DCE; record metrics, FLOPs, curve maps | Complete |
| `02_phase_a_diagnostics.ipynb` | Phase A Experiments 0–4 + Exp 3 vs 4 validation | Complete |
| `03_phase_b_experiments.ipynb` | Phase B: decoupling noise gain from contrast gain | **Not yet run** |
| `99_archive_superseded.ipynb` | Laplacian proxy diagnostics, `W_noise` sweeps | Abandoned — record only |

## How these are structured

Each notebook opens with a **single bootstrap cell** that pulls the repo once, mounts
Drive, loads the checkpoint, and defines every global the later cells use. It ends with
a smoke test printing `BOOTSTRAP OK` / `PHASE B READY`.

**If that smoke test fails, stop.** Nothing downstream is meaningful.

Experiment cells only read globals — they never define new ones. That rule exists
because the original notebook broke exactly this way; see `../CLAUDE.md`.

## Expected outputs

Markdown headers carry the expected numbers so a divergence is obvious. Key ones:

- **Exp 0** — reconstruction MAE 0; mean gain 1.3157 → 1.1626
- **Exp 2/3** — frozen amplification 78.35 / 82.65 / 104.22
- **Exp 4** — mean cumulative `J²` 76.988 at iteration 8; p99 293.82
- **Validation** — 1.59% / 6.60% / 31.46% relative difference
- **Exp 5.0** — `max_abs_error` ~`1e-7`; above `1e-5` means the decomposition is wrong

## Archive warning

`99_archive_superseded.ipynb` references runtime state that no longer exists; several
cells will not run. That is expected. Do not reuse its results or its `W_noise` weights
— `CLAUDE.md` explains why the objective was wrong, not merely mistuned.
