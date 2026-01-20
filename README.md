# Project-Catalyst

Single entry point: `main.py`. It runs all analyses and produces all plots with clear console progress. Optional: `fitting.py` fits a Gamma distribution to `data.csv` to supply realistic parameters.

**Goal**
- Study voting power in weighted voting games via Monte Carlo Banzhaf.
- Compare normalized Banzhaf to normalized weights across quotas and settings.

**Run**
- Requirements: Python 3.9+, `numpy`, `matplotlib`.
- Optional for fitting: `pandas`, `scipy` (for `fitting.py`).
- Quick start:
  - `python main.py`
- Optional (fit Gamma to data):
  - Place raw CSV at `data.csv` with columns `stake_address,voting_power` (header row allowed).
  - `python fitting.py` → prints `alpha` (shape) and `theta` (scale) MLE with `loc=0`.
- Outputs land in a single superfolder `plots2/` with subfolders:
  - `curves_default/` (default n combined curves and first-agent plots)
  - `weights_var/<range>/` (two-panel curves)
  - `bars/<range>/` (bar summaries)
  - `large_n/<name>/` (large-n variants per n across all three quota ranges)

What `main.py` does and outputs
- Parameters and grids
  - Parameter settings: list of Gamma shape/scale pairs; first entry `(0.24, 330235.0)` comes from the included real‑data fit path.
  - Quota grids: `full_0_1_101` (0..1), `focus_0p05_0p25_41` (0.05..0.25), `focus_0p4_0p6_41` (0.4..0.6).
  - Size knobs: `n` (agents), `M` (draws), `rp` (coalitions per draw).
- For each `(α, θ)` and each quota grid, the script:
  1) Samples weights `W ~ Gamma(α, θ)` and normalizes to `w_norm`.
  2) Estimates Banzhaf pivot probabilities across all quotas using `rp` random coalitions.
  3) Normalizes Banzhaf across players per quota, forms ratios `bn_i/w_i`.
  4) Aggregates over players and across `M` draws to compute means, variances, and 95% CIs.
  5) Saves the following plots:
    - `plots2/curves_default/alpha_*_combined_curve.pdf`: mean and variance of `bn_i/w_i` vs. quota (two y‑axes) with 95% CIs.
    - `plots2/weights_var/<range>/alpha_*_combined_curve_with_wvar.pdf`: two panels — top same as combined; bottom shows mean and 95% CI of `Var(w_norm)` across draws.
    - `plots2/bars/<range>/alpha_*_combined_bars.pdf`: bar summaries per quota of mean and variance of `bn_i/w_i` with 95% CIs.
    - `plots2/curves_default/alpha_*_first_agent_variance_curve.pdf`: variance over draws of `bn_1(q)/w_1` vs. quota with 95% CI.
    - `plots2/curves_default/alpha_*_first_agent_variance_boxplot.pdf`: boxplots of the variance estimator at representative quotas over `REPEATS` repeats.
    - `plots2/curves_default/alpha_*_first_agent_w1norm_boxplot.pdf`: boxplot of the first agent’s normalized weight across the `M` draws.

**Code Map (in `main.py`)**
- `draw_weights_gamma(n, alpha, theta, rng)`: samples `n` weights from `Gamma(α, θ)` via the provided `rng` and returns a vector `W`.
- `mc_banzhaf_all_quota_vectorized(W, rp, q_ratios, rng)`: vectorized Monte Carlo Banzhaf estimator across a quota grid.
  - Builds a random coalition matrix `T∈{0,1}^{rp×n}`; computes `others_sum` and `include_sum` per player.
  - For each quota `q = ratio · sum(W)`, marks pivots where `(others_sum < q) & (include_sum ≥ q)` and averages over coalitions to get pivot probabilities.
  - Returns `b∈R^{n×Q}` (players × quotas).
- `run_combined_curve(...)`: runs `M` draws; for each draw computes normalized Banzhaf per quota, divides by `w_norm` to get ratios; aggregates mean/variance curves with 95% CIs (SE over draws) and plots mean ratio (left axis) and variance (right axis).
- `run_curve_with_wvar(...)`: like combined curve, plus a lower panel with the variance of normalized weights `Var(w_norm)` with a 95% CI across draws to contextualize input inequality.
- `run_bars(...)`: bar summaries per quota for mean ratio and variance with CI error bars; x‑tick labels show the actual quotas.
- `run_first_agent(...)`: three diagnostics focused on agent 1.
  - Variance curve across quotas: computes the sample variance of `bn_1(q)/w_1` across `M` draws; CI for variance uses `s²·sqrt(2/(M−1))` (normal approximation), clipped at 0.
  - Multi‑quota boxplots over `REPEATS` independent experiments at selected quotas (`REPR_QS`).
  - Single‑agent weight boxplot over `M` draws to visualize `w_1` distribution under the Gamma input.
- Main loop: for each `(α, θ)` it generates a default combined curve, extended variants (weights‑var and bars) for each quota grid, several large‑`n` variants, and the first‑agent diagnostics. Master seed `42` ensures reproducibility; each parameter set derives its own child seed.

**CI/CD**
- GitHub Actions workflow at `./.github/workflows/ci.yml`:
  - Triggers on pushes to `main`/`master` and on PRs.
  - Sets up Python 3.10, installs `numpy` and `matplotlib`, and runs `flake8` (ignores E501).
  - Current smoke step imports `main_all`; update to import `main` if you switch to this filename in CI.
  - Extend as needed (tests, caching, pinned versions).

Behavior note for extreme gamma fit
- When using the real‑data gamma fit `(shape=0.24, scale=330235)`, sampled weights are highly skewed. After normalization, a single agent can dominate, leading to quotas where no coalition pivot events occur. In such cases, the Banzhaf column is all zeros and plots annotate “some quotas: no pivots”. Bars for those quotas may appear empty because the estimated mean/variance of `bn_i/w_i` is zero across all `M` draws.
- Mitigations: increase `M` and `rp` for this parameter and/or focus on quota ranges where pivots are more likely (often near 0.4–0.6 of total weight). Axes are clamped to the provided `q_ratios` range so focused plots (e.g., 0.05–0.25) display correct x‑limits.

**Key Settings (edit in `main.py`)**
This section explains the core ideas that influence runtime and interpretation.

Draws (M) vs. Repeats (REPEATS)
- Draws (M): Independent Monte Carlo replications used to build the main curves (combined, weights‑variance, bars) across every quota in the grid. Each draw re‑samples weights, estimates Banzhaf across all quotas, computes `bn_i/w_i`, and aggregates across players. Aggregating over `M` yields the mean and variance curves with 95% CI bands.
- Repeats (REPEATS): Only used for the first‑agent diagnostic box plots at a subset of representative quotas. For each chosen quota, a repeat runs `M` draws at that single quota, computes the sample variance of `bn_1(q)/w_1` across those `M` values, and records that one variance number. Repeating this `REPEATS` times builds a distribution that shows how variable the variance estimator itself is across independent runs.

Why use a subset of quotas for repeats
- Scientific coverage with tractable cost: we choose representative quotas to capture low (0.10, 0.20), center (0.50), and mid‑high (0.40, 0.60) regimes. Running repeats for all 101 quotas would be prohibitively slow; the subset offers meaningful insight at a fraction of the compute.

Performance guidance
- For previews, keep `M` and `REPEATS` modest (e.g., `M≈20`, `REPEATS≈20`) and optionally reduce `rp`; this accelerates iteration.
- For final runs, increase `M` (e.g., `50–100+`) and `rp` (e.g., `10k+`) to tighten CI bands; `REPEATS` can remain modest unless you specifically need detailed diagnostics.

How it works (at a glance)
- For each `(α, θ)` and quota grid:
  1) Sample weights: `W ~ Gamma(α, θ)`, normalize to `w_norm`.
  2) Estimate Banzhaf via `rp` random coalitions across all quotas.
  3) Normalize across players per quota; compute ratios `bn_i/w_i`.
  4) Aggregate over players and `M` draws to build means, variances, and 95% CIs.
  5) Save plots under `plots2/` with descriptive filenames.

**Console Progress**
- `main.py` prints global progress for each plot job, e.g., `[7/22] Large-n Bars (n1000_focus_0p4_0p6_31)`.
- Per‑plot, it prints periodic counters while iterating over `M` draws (and `REPEATS` repeats for the first‑agent diagnostics).

**Tips**
- Reduce `M` and/or `rp` to speed up iteration while testing visual choices.
- Re‑enable more `(alpha, theta)` pairs once satisfied with the layout.

Interpretation Guide
- Mean curve (`bn_i/w_i`): Values near 1 mean normalized Banzhaf aligns with normalized weights at that quota. Above 1 → agents tend to have more relative power than their weight; below 1 → less.
- Variance curve (`bn_i/w_i`): Peaks mark quotas where relative power is most dispersed across agents. Often higher near pivotal thresholds (around 0.5), reflecting sensitivity to coalition structure.
- Weight variance (`Var(w_norm)`): Higher values indicate more unequal weights. When weight inequality is extreme, some quotas can have no pivot events (annotated as “some quotas: no pivots”), yielding zero bars or flat segments.
- Bars vs. curves: Curves reveal trends across quotas with CIs; bars summarize per‑quota values with CIs. Use curves for pattern discovery and bars for side‑by‑side comparison at specific grids.
- Focused ranges: In `focus_0p05_0p25` and `focus_0p4_0p6`, x‑axes are clamped to the provided range to zoom into subtle or pivotal regions.
- First‑agent diagnostics:
  - Variance curve: Shows how unstable `bn_1(q)/w_1` is across draws for each quota; peaks = high sensitivity for that agent.
  - Multi‑quota boxplots: Distribution of the variance estimator over repeated experiments; wide boxes imply the estimator itself varies noticeably across independent runs.
- Extreme real‑data Gamma (`α=0.24, θ=330235`):
  - Expect very skewed weights; one agent may dominate.
  - Some quotas yield no pivots; plots annotate this. Consider focusing on 0.4–0.6 and increasing `M/rp` for more stable estimates.

Appendix: knobs and progress
- Key knobs at the bottom of `main.py` (main block):
  - `default_n` (100), `M` (20), `rp` (10000)
  - Quota sets: `full_0_1_101`, `focus_0p05_0p25_41`, `focus_0p4_0p6_41`
  - Large‑`n` configs: both `n=500` and `n=1000` across all three ranges
- Console progress shows total plots completed and per‑plot counters for draws and repeats.

