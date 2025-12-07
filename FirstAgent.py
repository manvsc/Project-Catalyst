import os
import numpy as np
import matplotlib.pyplot as plt


def draw_weights_gamma(n, alpha, theta):
    return np.random.gamma(shape=alpha, scale=theta, size=n)


def mc_banzhaf_all_quota_vectorized(W, rp, q_ratios, rng=None):
    """
    W: vector of weights (floats, length n)
    rp: number of coalition samples
    q_ratios: array of quota fractions in [0,1]
    Returns: b (n x Q) array of Banzhaf pivot probabilities
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(W)
    total = np.sum(W)
    quotas = q_ratios * total

    # sample rp coalitions uniformly
    T = rng.integers(0, 2, size=(rp, n), dtype=np.int8)  # (rp, n)
    W_mat = T * W
    sum_total = W_mat.sum(axis=1)                        # (rp,)
    others_sum = sum_total[:, None] - W_mat              # (rp, n)

    Q = len(q_ratios)
    b = np.zeros((n, Q))
    for j, q in enumerate(quotas):
        include_sum = others_sum + W
        pivots = (others_sum < q) & (include_sum >= q)
        b[:, j] = pivots.mean(axis=0)
    return b  # shape (n, Q)


def run_first_agent_variance(alpha, theta, n, M, rp, out_dir, rng, q_ratios):
    """
    For each draw m=1..M:
      - Sample weights W ~ Gamma(α, θ)
      - Compute normalized weights w_norm
      - Estimate Banzhaf pivot probs b_grid (n x Q)
      - Normalize across players per quota to get bn_grid
      - Compute ratio for player 0 only: ratio_0(q) = bn_0(q) / w_norm_0
    We then compute, across the M draws, the variance over draws of ratio_0(q) for each quota q
    and plot this variance curve with 95% CI bands (via SE across draws of the variance estimator).
    """
    tag = f"alpha_{str(alpha).replace('.', 'p')}_theta_{str(theta).replace('.', 'p')}"
    out_path = os.path.join(out_dir, f"{tag}_first_agent_variance_curve.pdf")
    boxplot_path = os.path.join(out_dir, f"{tag}_first_agent_variance_boxplot.pdf")

    Q = len(q_ratios)

    # Store the first agent's ratio per draw and quota: (M, Q)
    ratios_first_MQ = np.zeros((M, Q), dtype=float)

    for m in range(M):
        W = draw_weights_gamma(n, alpha, theta)
        w_norm = W / np.sum(W)

        b_grid = mc_banzhaf_all_quota_vectorized(W, rp=rp, q_ratios=q_ratios, rng=rng)

        # Normalize Banzhaf across players for each quota (columns sum to 1)
        col_sums = b_grid.sum(axis=0, keepdims=True)
        bn_grid = np.divide(b_grid, col_sums, out=np.zeros_like(b_grid), where=col_sums != 0)

        # Ratio for player 0 across quotas
        ratios_first_MQ[m, :] = bn_grid[0, :] / w_norm[0]

    # Across draws, compute variance of ratio for the first player at each quota
    # var_over_M(q) = Var_m[ ratios_first_MQ[m, q] ]
    # Also compute 95% CI for this variance estimate using SE across draws of the per-draw values squared.
    # We'll use the standard error of the sample variance via normal approximation:
    # If s2 is sample variance over M draws, Var(s2) ≈ (μ4 - ((M-3)/(M-1)) * s2^2) / M, but μ4 unknown.
    # As a simpler, robust display, we compute the SE of the variance estimates by bootstrap-like normal approx
    # using the delta method via sample fourth central moment.

    X = ratios_first_MQ  # (M, Q)
    means = X.mean(axis=0)
    centered = X - means
    s2 = (centered**2).sum(axis=0) / (M - 1) if M > 1 else np.zeros(Q)

    if M > 3:
        m4 = (centered**4).sum(axis=0) / (M - 1)
        # Approx variance of sample variance (unbiased s2) for normal data: Var(s2) ≈ 2*s2^2/(M-1)
        # Using sample fourth moment to be a bit more general:
        var_s2 = (m4 - ((M - 3) / (M - 1)) * (s2**2)) / M
        se_s2 = np.sqrt(np.maximum(var_s2, 0))
    elif M > 1:
        # Fallback normal-data approximation
        se_s2 = np.sqrt(2 * (s2**2) / (M - 1))
    else:
        se_s2 = np.zeros(Q)

    lo = s2 - 1.96 * se_s2
    hi = s2 + 1.96 * se_s2

    # Plot
    plt.figure()
    plt.plot(q_ratios, s2, color='purple', label='Var over draws of ratio (player 1)')
    plt.fill_between(q_ratios, lo, hi, color='purple', alpha=0.2, label='95% CI')
    plt.xlabel('quota')
    plt.ylabel('Variance of bn_1/w_1 over draws')
    plt.title(f"Gamma(α={alpha}, θ={theta}) — First agent ratio variance")
    plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    # Additionally: repeat experiment for first agent 100 times at representative quotas
    # to compare distributions across ranges. We pick: q in {0.10, 0.20, 0.50, 0.40, 0.60} (dedup if outside grid).
    repr_qs = [0.10, 0.20, 0.50, 0.40, 0.60]
    q_indices = []
    q_labels = []
    for q in repr_qs:
        idx = int(np.argmin(np.abs(q_ratios - q)))
        if (len(q_indices) == 0) or (idx != q_indices[-1]):
            q_indices.append(idx)
            q_labels.append(f"q={q_ratios[idx]:.2f}")

    repeats = 100
    var_samples_list = []
    for q_idx in q_indices:
        var_samples = np.zeros(repeats)
        for r in range(repeats):
            vals = np.zeros(M)
            for m in range(M):
                W = draw_weights_gamma(n, alpha, theta)
                w_norm = W / np.sum(W)
                b_grid = mc_banzhaf_all_quota_vectorized(W, rp=rp, q_ratios=q_ratios, rng=rng)
                col_sums = b_grid.sum(axis=0, keepdims=True)
                bn_grid = np.divide(b_grid, col_sums, out=np.zeros_like(b_grid), where=col_sums != 0)
                vals[m] = bn_grid[0, q_idx] / w_norm[0]
            var_samples[r] = np.var(vals, ddof=1) if M > 1 else 0.0
        var_samples_list.append(var_samples)

    # Multi-violin with overlaid compact box summaries
    plt.figure(figsize=(max(6, 1.8*len(q_indices)+2), 5))
    parts = plt.violinplot(var_samples_list, showmeans=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#87cefa')
        pc.set_alpha(0.5)
    plt.boxplot(var_samples_list, widths=0.2, positions=np.arange(1, len(q_indices)+1), vert=True,
                patch_artist=True, boxprops=dict(facecolor='lightsteelblue', alpha=0.7))
    plt.ylabel('Variance over M of bn_1/w_1 (first agent)')
    plt.xticks(np.arange(1, len(q_indices)+1), q_labels)
    plt.title(f"Gamma(α={alpha}, θ={theta}) — First agent variance across {repeats} repeats at multiple quotas")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(boxplot_path)
    plt.close()


if __name__ == "__main__":
    # Configuration similar to main.py, but focusing on first agent variance
    n = 100     # number of agents
    M = 100     # number of draws per (alpha, theta)
    rp = 10000  # number of coalition samples per draw
    PLOT_DIR = "plots/"
    os.makedirs(PLOT_DIR, exist_ok=True)

    # quota ratios from 0 to 1
    q_ratios = np.linspace(0, 1, 101)

    # Choose parameter settings to run
    # Use same parameter set as in main.py
    param_settings = [
        (0.5, 0.5),
        # (0.5, 1.0), (0.5, 2.0),
        # (1.0, 0.5), (1.0, 1.0), (1.0, 2.0),
        # (2.0, 0.5), (2.0, 1.0), (2.0, 2.0),
        # (5.0, 1.0),
    ]

    master_rng = np.random.default_rng(42)

    for idx, (alpha, theta) in enumerate(param_settings, start=1):
        print(f"[FirstAgent {idx}/{len(param_settings)}] alpha={alpha}, theta={theta}")
        rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
        run_first_agent_variance(alpha, theta, n, M, rp, PLOT_DIR, rng, q_ratios)

    print(f"Done. First-agent variance plots saved in: {PLOT_DIR}")
