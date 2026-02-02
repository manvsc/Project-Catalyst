# Project-Catalyst

Single entry point: `main.py`.

**Goal**
- Study fairness in weighted voting games via Monte Carlo Banzhaf.
- Compare **normalized Banzhaf** to **normalized weights** across quotas, user counts, and Gamma parameters.

**Run**
- Requirements: Python 3.9+, `numpy`, `matplotlib`.
- Optional for fitting: `pandas`, `scipy` (for `fitting2.py`).
- Quick start:
  - `python main.py`

**Fitting**
- `fitting2.py` fits a Gamma distribution to `data2.csv` (filtered to `voted == True`).
- It prints `alpha` (shape) and `theta` (scale) with `loc=0`.
- The current fitted values used by `main.py` are:
  - `alpha = 0.273568`
  - `theta = 1301506.236646`

**What the ratios are**
- For each quota, the code computes:
  - `bn_i(q)` = Banzhaf pivot probability for player `i`.
  - `bn_i(q)` is normalized across players so `sum_i bn_i(q) = 1`.
  - `w_i` is normalized weight so `sum_i w_i = 1`.
  - The ratio is **(normalized Banzhaf) / (normalized weight)**.

**Intergroup mean and variance (how computed)**
- For each Monte Carlo draw:
  1) Sample weights `W ~ Gamma(alpha, theta)` for `n` users.
  2) Estimate Banzhaf `bn_i(q)` for every user and quota using `rp` random coalitions.
  3) Normalize Banzhaf across users per quota and compute ratios `r_i(q) = bn_i(q) / w_i`.
  4) Compute **intergroup mean** per quota: `mean_r(q) = (1/n) * sum_i r_i(q)`.
  5) Compute **intergroup variance** per quota across users (sample variance, `ddof=1`):  
     `var_r(q) = (1/(n-1)) * sum_i (r_i(q) - mean_r(q))^2`.
- The plotted curves and boxplots summarize these intergroup means/variances **across M independent draws**:
  - Mean curves = average of `mean_r(q)` across draws.
  - Variance curves = average of `var_r(q)` across draws.
  - Boxplots = distribution of `mean_r(q)` or `var_r(q)` across draws at fixed quota.

**Outputs (all under `plots2/intergroup_variance/`)**
1) **Multi-n mean + variance curves (intergroup)**  
   - `alpha_*_theta_*_mean_and_variance_multi_n.pdf`
2) **Multi-n mean + variance curves (first agent)**  
   - `alpha_*_theta_*_first_agent_mean_and_variance_multi_n.pdf`
3) **Fixed-quota boxplots (mean + variance) across n**  
   - `alpha_*_theta_*_intergroup_mean_and_variance_boxplot_q0.070.pdf`
4) **Multi-parameter mean + variance curves (n=100)**  
   - `mean_and_variance_multi_params_n100.pdf`

**Parameter settings for multi-parameter plot**
- Baseline: `(0.5, 0.5)`
- Fitted: `(alpha_fit, theta_fit)`
- Vary theta with fitted alpha:
  - `(alpha_fit, 0.5 * theta_fit)`
  - `(alpha_fit, 2.0 * theta_fit)`
- Vary alpha with fitted theta:
  - `(0.5 * alpha_fit, theta_fit)`
  - `(2.0 * alpha_fit, theta_fit)`

**Key settings (edit in `main.py`)**
- `default_n`: default number of users for fixed-n plots (currently 100).
- `intergroup_n_list`: user counts for multi-n plots (currently `[20, 50, 100]`).
- `M`: number of Monte Carlo draws.
- `rp`: coalitions per draw.
- `intergroup_q_fixed`: fixed quota (currently `0.07`).
- `quota_sets['full_0_1_101']`: quota grid used for curves.

**Reproducibility**
- All randomness is derived from a master seed (`master_seed = 42`).
- Each plot uses a deterministic RNG seeded by a tagged hash, so results are reproducible and independent of run order.

**Notes on axes**
- Variance axes are clamped to start at 0, with a small visual pad below zero (ticks stay non-negative).
- Some variance panels use log scale when the linear scale would hide smaller curves.

**How it works (at a glance)**
1) Sample weights `W ~ Gamma(alpha, theta)` and normalize to `w_norm`.
2) Estimate Banzhaf pivot probabilities via random coalitions.
3) Normalize Banzhaf per quota; compute ratios `bn_i / w_i`.
4) Aggregate over players and `M` draws to compute means and variances.
