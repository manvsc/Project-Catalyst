# Project-Catalyst

Single entry point: main_all.py. It runs all analyses and produces all plots, with clear console progress.

**Goal**
- Study voting power in weighted voting games via Monte Carlo Banzhaf.
- Compare normalized Banzhaf to normalized weights across quotas and settings.

**Run**
- Requirements: Python 3.9+, `numpy`, `matplotlib`.
- Quick start:
  - `python main_all.py`
- Outputs land in a single superfolder `plots/` with subfolders:
  - `curves_default/` (default n combined curves and first-agent plots)
  - `weights_var/<range>/` (two-panel curves)
  - `bars/<range>/` (bar summaries)
  - `large_n/<name>/` (large-n variants per n across all three quota ranges)

What `main_all.py` does and outputs
- Parameters and grids
  - Parameter settings: list of Gamma shape/scale pairs; first entry `(0.24, 330235.0)` comes from a real-data fit.
  - Quota grids: `full_0_1_101` (0..1), `focus_0p05_0p25_41` (0.05..0.25), `focus_0p4_0p6_41` (0.4..0.6).
  - Size knobs: `n` (agents), `M` (draws), `rp` (coalitions per draw).
- For each `(α, θ)` and each quota grid, the script:
  1) Samples weights `W ~ Gamma(α, θ)` and normalizes to `w_norm`.
  2) Estimates Banzhaf pivot probabilities across all quotas using `rp` random coalitions.
  3) Normalizes Banzhaf across players per quota, forms ratios `bn_i/w_i`.
  4) Aggregates over players and across `M` draws to compute means, variances, and 95% CIs.
  5) Saves the following plots:
    - `plots/curves_default/alpha_*_combined_curve.pdf`: mean and variance of `bn_i/w_i` vs. quota (two y-axes) with 95% CIs.
    - `plots/weights_var/<range>/alpha_*_combined_curve_with_wvar.pdf`: two panels — top same as combined; bottom shows mean and 95% CI of `Var(w_norm)` across draws.
    - `plots/bars/<range>/alpha_*_combined_bars.pdf`: bar summaries per quota of mean and variance of `bn_i/w_i` with 95% CIs.
    - `plots/curves_default/alpha_*_first_agent_variance_curve.pdf`: variance over draws of `bn_1(q)/w_1` vs. quota with 95% CI.
    - `plots/curves_default/alpha_*_first_agent_variance_boxplot.pdf`: boxplots of the variance estimator at representative quotas over `REPEATS` repeats.
    - `plots/curves_default/alpha_*_first_agent_w1norm_boxplot.pdf`: boxplot of the first agent’s normalized weight across the `M` draws.

Behavior note for extreme gamma fit
- When using the real-data gamma fit `(shape=0.24, scale=330235)`, sampled weights are highly skewed. After normalization, a single agent can dominate, leading to quotas where no coalition pivot events occur. In such cases, the Banzhaf column is all zeros, and plots annotate “some quotas: no pivots”. Bars for those quotas may appear empty because the estimated mean/variance of `bn_i/w_i` is zero across all M draws.
- Mitigations: increase `M` and `rp` for this parameter and/or focus on quota ranges where pivots are more likely (often near 0.4–0.6 of total weight). The code clamps x-axes to the provided `q_ratios` range so focused plots (e.g., 0.05–0.25) display correct x-limits.

**Key Settings (edit in main_all.py)**
This section explains the core ideas that influence runtime and interpretation.

Draws (M) vs. Repeats (REPEATS)
- Draws (M): Independent Monte Carlo replications used to build the main curves (combined, weights-variance, bars) across every quota in the grid. Each draw re-samples weights, estimates Banzhaf across all quotas, computes bn_i/w_i, and aggregates across players. Aggregating over M yields the mean and variance curves with 95% CI bands.
- Repeats (REPEATS): Only used for the first-agent diagnostic violin/box plots at a subset of representative quotas. For each chosen quota, a repeat runs M draws at that single quota, computes the sample variance of bn_1(q)/w_1 across those M values, and records that one variance number. Repeating this REPEATS times builds a distribution that shows how variable the variance estimator itself is across independent runs.

Why use a subset of quotas for repeats
- Scientific coverage with tractable cost: we choose representative quotas to capture low (0.10, 0.20), center (0.50), and mid-high (0.40, 0.60) regimes. Running repeats for all 101 quotas would be prohibitively slow; the subset offers meaningful insight at a fraction of the compute.

Performance guidance
- For previews, keep M and REPEATS modest (e.g., M≈20, REPEATS≈20) and optionally reduce rp; this accelerates iteration.
- For final runs, increase M (e.g., 50–100+) and rp (e.g., 10k+) to tighten CI bands; REPEATS can remain modest unless you specifically need detailed violin diagnostics.

How it works (at a glance)
- For each `(α, θ)` and quota grid:
  1) Sample weights: `W ~ Gamma(α, θ)`, normalize to `w_norm`.
  2) Estimate Banzhaf via `rp` random coalitions across all quotas.
  3) Normalize across players per quota; compute ratios `bn_i/w_i`.
  4) Aggregate over players and `M` draws to build means, variances, and 95% CIs.
  5) Save plots under `plots/` with descriptive filenames.

**Console Progress**
- main_all.py prints global progress for each plot job, e.g., `[7/22] bars n=100 focus_0p4_0p6_41 a=0.5 t=0.5`.
- Per-plot, it prints periodic counters while iterating over `M` draws (and 100 repeats for the first-agent violin).

**Tips**
- Reduce `M` and/or `rp` to speed up iteration while testing visual choices.
- Re-enable more `(alpha, theta)` pairs once satisfied with the layout.

Appendix: knobs and progress
- Key knobs at the top of `main_all.py`:
  - `default_n` (default 100), `M` (default 20 by default in this repo), `rp` (default 10000)
  - Quota sets: `full_0_1_101`, `focus_0p05_0p25_41`, `focus_0p4_0p6_41`
  - Large-`n` configs: both n=500 and n=1000 across all three ranges
- Console progress shows total plots completed and per-plot counters for draws and repeats.
