import numpy as np
import os
import matplotlib.pyplot as plt

# -----------------------------
# Draw random weights directly from Gamma
# -----------------------------
def draw_weights_gamma(n, alpha, theta):
    return np.random.gamma(shape=alpha, scale=theta, size=n)

# -----------------------------
# Monte Carlo Banzhaf for all players & quotas (float weights)
# -----------------------------
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
    Q = len(q_ratios)
    total = np.sum(W)
    quotas = q_ratios * total

    # sample rp coalitions uniformly
    T = rng.integers(0, 2, size=(rp, n), dtype=np.int8)  # (rp, n)
    W_mat = T * W
    sum_total = W_mat.sum(axis=1)                        # (rp,)
    others_sum = sum_total[:, None] - W_mat              # (rp, n)

    b = np.zeros((n, Q))
    for j, q in enumerate(quotas):
        include_sum = others_sum + W
        pivots = (others_sum < q) & (include_sum >= q)
        b[:, j] = pivots.mean(axis=0)
    return b  # shape (n, Q)

# -----------------------------
# Run one Gamma setting
# -----------------------------
def run_setting(alpha, theta, n, M, rp, out_dir, rng, q_ratios):
    tag = f"alpha_{str(alpha).replace('.','p')}_theta_{str(theta).replace('.','p')}"
    combined_plot = os.path.join(out_dir, f"{tag}_combined_curve.pdf")

    Q = len(q_ratios)
    means_MQ = np.zeros((M, Q), dtype=float)
    vars_MQ  = np.zeros((M, Q), dtype=float)

    for m in range(M):
        W = draw_weights_gamma(n, alpha, theta)
        w_norm = W / np.sum(W)

        # Raw Banzhaf pivot probabilities per player and quota
        b_grid = mc_banzhaf_all_quota_vectorized(W, rp=rp, q_ratios=q_ratios, rng=rng)
        # Normalize Banzhaf across players for each quota (columns sum to 1)
        col_sums = b_grid.sum(axis=0, keepdims=True)
        bn_grid = np.divide(b_grid, col_sums, out=np.zeros_like(b_grid), where=col_sums != 0)
        # Ratio of normalized Banzhaf to normalized weights
        ratios = bn_grid / w_norm[:, None]
        means_MQ[m, :] = ratios.mean(axis=0)
        vars_MQ[m, :]  = ratios.var(axis=0)

    # Aggregate across draws (for plotting confidence bands)
    def ci95(samples):
        samples = np.asarray(samples, dtype=float)
        m = float(np.mean(samples))
        se = float(np.std(samples, ddof=1) / np.sqrt(len(samples))) if len(samples) > 1 else 0.0
        return m, (m - 1.96 * se, m + 1.96 * se), float(np.var(samples, ddof=1)) if len(samples) > 1 else 0.0

    # --- Combined plot with two y-axes ---
    mean_of_means = means_MQ.mean(axis=0)
    se_means = means_MQ.std(axis=0, ddof=1) / np.sqrt(M) if M > 1 else np.zeros(Q)
    lo_mean = mean_of_means - 1.96 * se_means
    hi_mean = mean_of_means + 1.96 * se_means

    mean_of_vars = vars_MQ.mean(axis=0)
    se_vars = vars_MQ.std(axis=0, ddof=1) / np.sqrt(M) if M > 1 else np.zeros(Q)
    lo_var = mean_of_vars - 1.96 * se_vars
    hi_var = mean_of_vars + 1.96 * se_vars

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Left y-axis: mean-of-means
    ax1.plot(q_ratios, mean_of_means, color='blue', label='Mean of means')
    ax1.fill_between(q_ratios, lo_mean, hi_mean, color='blue', alpha=0.2)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1)   # reference at 1
    ax1.set_ylabel("Mean of (bn_i / w_i)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Right y-axis: mean-of-vars
    ax2.plot(q_ratios, mean_of_vars, color='red', linestyle='--', label='Mean of vars')
    ax2.fill_between(q_ratios, lo_var, hi_var, color='red', alpha=0.2)
    ax2.set_ylabel("Variance of (bn_i / w_i)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax1.set_xlabel("quota")
    ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)  # reference line at 0.5
    plt.title(f"Gamma(α={alpha}, θ={theta}) — normalized Banzhaf ratios")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(combined_plot)
    plt.close()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    n = 100     # number of agents
    M = 100      # number of draws per (alpha, theta)
    rp = 10000    # number of coalition samples per draw
    PLOT_DIR = "plots/"
    os.makedirs(PLOT_DIR, exist_ok=True)

    # quota ratios from 0 to 1
    q_ratios = np.linspace(0, 1, 101)

    param_settings = [
        (0.5, 0.5),
        (0.5, 1.0),
        (0.5, 2.0),
        (1.0, 0.5),
        (1.0, 1.0),
        (1.0, 2.0),
        (2.0, 0.5),
        (2.0, 1.0),
        (2.0, 2.0),
        (5.0, 1.0),
    ]

    master_rng = np.random.default_rng(42)

    for idx, (alpha, theta) in enumerate(param_settings, start=1):
        print(f"[{idx}/{len(param_settings)}] alpha={alpha}, theta={theta}")
        rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
        run_setting(alpha, theta, n, M, rp, PLOT_DIR, rng, q_ratios)

    print(f"Done. Results saved in: {PLOT_DIR}")
