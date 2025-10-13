# Project-Catalyst

High-level overview, how to run, what the code does step-by-step, and annotated snippets.

**High-Level Idea**

- Study voting power in weighted voting games using Monte Carlo simulation of the Banzhaf power index.
- Draw agent weights from a Gamma distribution (parameters α, θ), sweep quota ratios `q ∈ [0, 1]`, and estimate per-player pivot probabilities.
- Normalize Banzhaf power across players for each quota and compare to normalized weights via the ratio `bn_i / w_i`.
- Aggregate across many independent draws to understand how power relates to weights across quotas, with confidence intervals, and save one plot per (α, θ) setting.

---

**What the Script Produces**

- One PDF per parameter pair `(alpha, theta)` in `plots/`, named like:
  - `alpha_0p5_theta_1p0_combined_curve.pdf`
- Each PDF is a combined figure with two y-axes over quota ratio `q`:
  - Left axis (blue): mean of the ratios `bn_i / w_i` across players, with 95% CI bands.
  - Right axis (red): variance of the ratios `bn_i / w_i` across players, with 95% CI bands.

---

**How to Run**

- Requirements: Python 3.9+, `numpy`, `matplotlib`.
- Quick start:
  - Option A (use your environment):
    - `python main.py`
  - Option B (fresh venv):
    - `python -m venv .venv && source .venv/bin/activate`
    - `pip install numpy matplotlib`
    - `python main.py`
- Outputs land in `plots/`.

---

**Configuration (in `main.py`)**

- `n`: number of agents (default 100)
- `M`: independent draws per parameter setting (default 100)
- `rp`: Monte Carlo coalitions per draw (default 1000)
- `q_ratios`: quota sweep (default 0.00 → 1.00 in 0.01 steps)
- `param_settings`: list of `(alpha, theta)` Gamma parameters to run
- `PLOT_DIR`: output directory (`plots/`)
- Reproducibility: master RNG seeded with `42`.

You can narrow `param_settings` to a single pair if you only want one plot.

---

**Step-by-Step: What the Code Does**

1) Sample weights
- For each `(alpha, theta)` and for `M` draws: sample `n` agent weights `W ~ Gamma(α, θ)` and compute normalized weights `w_norm = W / sum(W)`.

2) Monte Carlo Banzhaf across quotas
- For each draw, estimate pivot probabilities for all players simultaneously across all quotas in `q_ratios`:
  - Sample `rp` random coalitions uniformly (`0/1` membership for each player).
  - Compute coalition weight sums vectorized.
  - For each quota, a player is a pivot if excluding them is below quota, but including them reaches or exceeds quota.

3) Normalize and form ratios
- For each quota column, normalize Banzhaf values across players so they sum to 1 (comparable to normalized weights).
- Compute ratios `ratios_iq = bn_iq / w_i` for each player `i` and quota `q`.

4) Aggregate across players and draws
- For each draw and quota, compute `mean` and `variance` of `ratios` across players.
- Across `M` draws, compute mean-of-means and mean-of-variances, plus 95% CIs (via standard error across draws).

5) Plot and save
- Create a combined plot with two y-axes vs. `q` and save as a PDF per `(alpha, theta)` in `plots/`.

---

**Interpreting the Plot**

- If the blue curve (mean of ratios) is near 1, normalized Banzhaf power aligns with normalized weights at that quota.
- Deviations from 1 indicate systematic advantages/disadvantages relative to raw weights.
- The red curve (variance) shows dispersion of `bn_i / w_i` across players for the same quota (heterogeneity of relative power).

---

**Key Snippets Explained**

1) Drawing Gamma weights

```python
def draw_weights_gamma(n, alpha, theta):
    return np.random.gamma(shape=alpha, scale=theta, size=n)
```
- Samples `n` iid weights from `Gamma(α=alpha, θ=theta)`.

2) Vectorized Monte Carlo Banzhaf over quotas

```python
def mc_banzhaf_all_quota_vectorized(W, rp, q_ratios, rng=None):
    n = len(W)
    Q = len(q_ratios)
    total = np.sum(W)
    quotas = q_ratios * total

    # Sample rp coalitions uniformly in {0,1}^n
    T = rng.integers(0, 2, size=(rp, n), dtype=np.int8)
    W_mat = T * W
    sum_total = W_mat.sum(axis=1)           # (rp,)
    others_sum = sum_total[:, None] - W_mat  # (rp, n)

    b = np.zeros((n, Q))
    for j, q in enumerate(quotas):
        include_sum = others_sum + W
        pivots = (others_sum < q) & (include_sum >= q)
        b[:, j] = pivots.mean(axis=0)
    return b
```
- Samples random coalitions and checks pivot conditions for all players and quotas in one pass.
- `pivots.mean(axis=0)` yields pivot probability per player for quota `q`.

3) Normalization, ratios, and aggregation

```python
b_grid = mc_banzhaf_all_quota_vectorized(W, rp=rp, q_ratios=q_ratios, rng=rng)
col_sums = b_grid.sum(axis=0, keepdims=True)
bn_grid = np.divide(b_grid, col_sums, out=np.zeros_like(b_grid), where=col_sums != 0)
ratios = bn_grid / w_norm[:, None]

means_MQ[m, :] = ratios.mean(axis=0)
vars_MQ[m, :]  = ratios.var(axis=0)
```
- Normalize Banzhaf across players per quota to compare apples-to-apples with normalized weights.
- Record per-draw mean and variance of ratios across players for later CI computation.

4) Plotting the combined figure

```python
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(q_ratios, mean_of_means, color='blue')
ax1.fill_between(q_ratios, lo_mean, hi_mean, color='blue', alpha=0.2)
ax2.plot(q_ratios, mean_of_vars, color='red', linestyle='--')
ax2.fill_between(q_ratios, lo_var, hi_var, color='red', alpha=0.2)
plt.savefig(combined_plot)
```
- Left axis shows central tendency of `bn_i / w_i`; right axis shows dispersion.

---

**Tips & Troubleshooting**

- Runtime vs. accuracy: Increasing `rp` and `M` improves stability but costs time.
- If you only see one plot, ensure `param_settings` has multiple entries and the run completed (watch console progress `[i/N]`).
- For very large `n` or `rp`, consider chunking or profiling to manage memory/time.

---

**Extending**

- Replace the Gamma sampler with other weight distributions.
- Add alternative power indices (e.g., Shapley-Shubik via sampling) for comparison.
- Export summary CSVs per setting alongside plots.
