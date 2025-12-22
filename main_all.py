import os
import numpy as np
import matplotlib.pyplot as plt


# --- CORE UTILITIES ---

def draw_weights_gamma(n, alpha, theta):
    """Sample a length-n vector of agent weights from Gamma(α=alpha, θ=theta)."""
    return np.random.gamma(shape=alpha, scale=theta, size=n)


def mc_banzhaf_all_quota_vectorized(W, rp, q_ratios, rng=None):
    """Estimate Banzhaf pivot probabilities for all players across a quota grid.

    Args:
        W: array of agent weights (length n)
        rp: number of random coalitions to sample
        q_ratios: array of quota ratios in [0,1]; quotas = q_ratios * sum(W)
        rng: optional numpy Generator for reproducibility

    Returns:
        b: (n, Q) array of pivot probabilities for each player and quota.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(W)
    Q = len(q_ratios)
    total = np.sum(W)
    quotas = q_ratios * total

    # Generate random coalitions (0 or 1)
    T = rng.integers(0, 2, size=(rp, n), dtype=np.int8)
    W_mat = T * W
    sum_total = W_mat.sum(axis=1)
    others_sum = sum_total[:, None] - W_mat

    b = np.zeros((n, Q))
    for j, q in enumerate(quotas):
        include_sum = others_sum + W
        # Pivot condition: Losing without me (< q) AND Winning with me (>= q)
        pivots = (others_sum < q) & (include_sum >= q)
        b[:, j] = pivots.mean(axis=0)
    return b


# --- PLOTTING FUNCTIONS ---

def run_combined_curve(alpha, theta, n, M, rp, out_dir, rng, q_ratios):
    """Combined curve plot over quotas showing:
    - Left axis: Mean Power/Stake Ratio (Target = 1.0)
    - Right axis: Variance of Power/Stake Ratio (Target = 0.0)
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha_{str(alpha).replace('.', 'p')}_theta_{str(theta).replace('.', 'p')}"
    out_path = os.path.join(out_dir, f"{tag}_combined_curve.pdf")

    Q = len(q_ratios)
    means_MQ = np.zeros((M, Q))
    vars_MQ = np.zeros((M, Q))

    for m in range(M):
        if M >= 5 and (m == 0 or (m + 1) % max(1, M // 5) == 0 or m == M - 1):
            print(f"    [run_combined_curve] draw {m + 1}/{M}")
        W = draw_weights_gamma(n, alpha, theta)
        w_norm = W / np.sum(W)
        b_grid = mc_banzhaf_all_quota_vectorized(W, rp=rp, q_ratios=q_ratios, rng=rng)

        # Normalize Banzhaf scores so they sum to 1
        col_sums = b_grid.sum(axis=0, keepdims=True)
        bn_grid = np.divide(b_grid, col_sums, out=np.zeros_like(b_grid), where=col_sums != 0)

        # Calculate Ratio: (Normalized Banzhaf) / (Normalized Stake)
        ratios = bn_grid / w_norm[:, None]
        means_MQ[m, :] = ratios.mean(axis=0)
        vars_MQ[m, :] = ratios.var(axis=0)

    # --- STATISTICS & CLIPPING FIX ---
    mean_of_means = means_MQ.mean(axis=0)
    se_means = means_MQ.std(axis=0, ddof=1) / np.sqrt(M) if M > 1 else np.zeros(Q)
    # FIX: Clip lower bound at 0
    lo_mean = np.maximum(0, mean_of_means - 1.96 * se_means)
    hi_mean = mean_of_means + 1.96 * se_means

    mean_of_vars = vars_MQ.mean(axis=0)
    se_vars = vars_MQ.std(axis=0, ddof=1) / np.sqrt(M) if M > 1 else np.zeros(Q)
    # FIX: Clip lower bound at 0
    lo_var = np.maximum(0, mean_of_vars - 1.96 * se_vars)
    hi_var = mean_of_vars + 1.96 * se_vars

    # --- PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    # Left Axis (Mean)
    ax1.plot(q_ratios, mean_of_means, color='blue', label='Mean Ratio')
    ax1.fill_between(q_ratios, lo_mean, hi_mean, color='blue', alpha=0.2)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1, label="Perfect Proportionality (1.0)")
    ax1.set_ylabel("Mean Power/Stake Ratio\n(Closer to 1 is better)", color='blue', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='blue')

    # Right Axis (Variance)
    ax2.plot(q_ratios, mean_of_vars, color='red', linestyle='--', label='Variance of Ratio')
    ax2.fill_between(q_ratios, lo_var, hi_var, color='red', alpha=0.2)
    ax2.set_ylabel("Variance of Power/Stake Ratio\n(Lower is better)", color='red', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='red')

    ax1.set_xlabel("Quota (Proportion of Total Stake)")
    ax1.set_xlim(float(q_ratios.min()), float(q_ratios.max()))
    ax1.axvline(x=0.5, color='gray', linestyle=':', linewidth=1)

    # Self-Explanatory Title
    plt.title(
        f"Fairness Analysis: Power vs. Stake\nGamma Dist (Shape $\\alpha$={alpha}, Scale $\\theta$={theta}, n={n})",
        fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_curve_with_wvar(alpha, theta, n, M, rp, out_dir, rng, q_ratios):
    """Two-panel figure:
    - Top: Mean and Variance of Power/Stake Ratio (Corrected CIs)
    - Bottom: Variance of the Weights themselves (control metric)
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha_{str(alpha).replace('.', 'p')}_theta_{str(theta).replace('.', 'p')}"
    out_path = os.path.join(out_dir, f"{tag}_combined_curve_with_wvar.pdf")

    Q = len(q_ratios)
    means_MQ = np.zeros((M, Q))
    vars_MQ = np.zeros((M, Q))
    wvar_M = np.zeros(M)

    for m in range(M):
        if M >= 5 and (m == 0 or (m + 1) % max(1, M // 5) == 0 or m == M - 1):
            print(f"    [run_curve_with_wvar] draw {m + 1}/{M}")
        W = draw_weights_gamma(n, alpha, theta)
        w_norm = W / np.sum(W)
        wvar_M[m] = np.var(w_norm)
        b_grid = mc_banzhaf_all_quota_vectorized(W, rp=rp, q_ratios=q_ratios, rng=rng)
        col_sums = b_grid.sum(axis=0, keepdims=True)
        bn_grid = np.divide(b_grid, col_sums, out=np.zeros_like(b_grid), where=col_sums != 0)
        ratios = bn_grid / w_norm[:, None]
        means_MQ[m, :] = ratios.mean(axis=0)
        vars_MQ[m, :] = ratios.var(axis=0)

    # --- STATISTICS & CLIPPING ---
    mean_of_means = means_MQ.mean(axis=0)
    se_means = means_MQ.std(axis=0, ddof=1) / np.sqrt(M) if M > 1 else np.zeros(Q)
    lo_mean = np.maximum(0, mean_of_means - 1.96 * se_means)
    hi_mean = mean_of_means + 1.96 * se_means

    mean_of_vars = vars_MQ.mean(axis=0)
    se_vars = vars_MQ.std(axis=0, ddof=1) / np.sqrt(M) if M > 1 else np.zeros(Q)
    lo_var = np.maximum(0, mean_of_vars - 1.96 * se_vars)
    hi_var = mean_of_vars + 1.96 * se_vars

    wvar_mean = float(np.mean(wvar_M))
    wvar_se = float(np.std(wvar_M, ddof=1) / np.sqrt(M)) if M > 1 else 0.0
    wvar_ci_lo = max(0, wvar_mean - 1.96 * wvar_se)
    wvar_ci_hi = wvar_mean + 1.96 * wvar_se

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1b = ax1.twinx()

    # Top Panel
    ax1.plot(q_ratios, mean_of_means, color='blue', label="Mean Ratio")
    ax1.fill_between(q_ratios, lo_mean, hi_mean, color='blue', alpha=0.2)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax1.set_ylabel("Mean Power/Stake Ratio", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)

    ax1b.plot(q_ratios, mean_of_vars, color='red', linestyle='--', label="Var Ratio")
    ax1b.fill_between(q_ratios, lo_var, hi_var, color='red', alpha=0.2)
    ax1b.set_ylabel("Variance of Ratio", color='red')
    ax1b.tick_params(axis='y', labelcolor='red')

    ax1.set_xlabel("Quota")
    ax1.set_xlim(float(q_ratios.min()), float(q_ratios.max()))
    ax1.axvline(x=0.5, color='gray', linestyle=':', linewidth=1)
    ax1.set_title(f"Fairness Metrics vs. Quota (n={n})\nGamma Dist (Shape $\\alpha$={alpha}, Scale $\\theta$={theta})")

    # Bottom Panel (Control)
    ax2.bar([0], [wvar_mean], width=0.6, color='purple', alpha=0.8)
    ax2.errorbar([0], [wvar_mean], yerr=[[wvar_mean - wvar_ci_lo], [wvar_ci_hi - wvar_mean]],
                 fmt='none', ecolor='black', capsize=3)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Var(Normalized Weights)"])
    ax2.set_ylabel("Variance of Input Weights")
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_title("Control Metric: Inequality of Stake Distribution")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_bars(alpha, theta, n, M, rp, out_dir, rng, q_ratios):
    """Bar summary of mean and variance of bn_i/w_i across quotas with 95% CI bars."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha_{str(alpha).replace('.', 'p')}_theta_{str(theta).replace('.', 'p')}"
    out_path = os.path.join(out_dir, f"{tag}_combined_bars.pdf")

    Q = len(q_ratios)
    means_MQ = np.zeros((M, Q))
    vars_MQ = np.zeros((M, Q))
    for m in range(M):
        if M >= 5 and (m == 0 or (m + 1) % max(1, M // 5) == 0 or m == M - 1):
            print(f"    [run_bars] draw {m + 1}/{M}")
        W = draw_weights_gamma(n, alpha, theta)
        w_norm = W / np.sum(W)
        b_grid = mc_banzhaf_all_quota_vectorized(W, rp=rp, q_ratios=q_ratios, rng=rng)
        col_sums = b_grid.sum(axis=0, keepdims=True)
        bn_grid = np.divide(b_grid, col_sums, out=np.zeros_like(b_grid), where=col_sums != 0)
        ratios = bn_grid / w_norm[:, None]
        means_MQ[m, :] = ratios.mean(axis=0)
        vars_MQ[m, :] = ratios.var(axis=0)

    # --- STATISTICS & CLIPPING ---
    mean_of_means = means_MQ.mean(axis=0)
    se_means = means_MQ.std(axis=0, ddof=1) / np.sqrt(M) if M > 1 else np.zeros(Q)
    lo_mean = np.maximum(0, mean_of_means - 1.96 * se_means)
    hi_mean = mean_of_means + 1.96 * se_means

    mean_of_vars = vars_MQ.mean(axis=0)
    se_vars = vars_MQ.std(axis=0, ddof=1) / np.sqrt(M) if M > 1 else np.zeros(Q)
    lo_var = np.maximum(0, mean_of_vars - 1.96 * se_vars)
    hi_var = mean_of_vars + 1.96 * se_vars

    x = np.arange(Q)
    width = 0.6
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top Bar (Mean)
    ax_top.bar(x, mean_of_means, width=width, color='steelblue', alpha=0.8)
    ax_top.errorbar(x, mean_of_means, yerr=[mean_of_means - lo_mean, hi_mean - mean_of_means],
                    fmt='none', ecolor='black', elinewidth=1, capsize=2)
    ax_top.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax_top.set_ylabel("Mean Power/Stake Ratio\n(Target = 1.0)")
    ax_top.grid(True, axis='y', alpha=0.3)
    ax_top.set_title(f"Fairness Summary by Quota (n={n})\nGamma(Shape $\\alpha$={alpha}, Scale $\\theta$={theta})")

    # Bottom Bar (Variance)
    ax_bottom.bar(x, mean_of_vars, width=width, color='indianred', alpha=0.8)
    ax_bottom.errorbar(x, mean_of_vars, yerr=[mean_of_vars - lo_var, hi_var - mean_of_vars],
                       fmt='none', ecolor='black', elinewidth=1, capsize=2)
    ax_bottom.set_ylabel("Variance of Power/Stake Ratio\n(Target = 0.0)")
    ax_bottom.set_xlabel("Quota Index")
    ax_bottom.grid(True, axis='y', alpha=0.3)

    tick_idx = np.linspace(0, Q - 1, min(Q, 11), dtype=int)
    ax_bottom.set_xticks(tick_idx)
    ax_bottom.set_xticklabels([f"{q:.2f}" for q in q_ratios[tick_idx]])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path)
    plt.close()


def run_first_agent(alpha, theta, n, M, rp, out_dir, rng, q_ratios, repr_qs=None, repeats=20):
    """First-agent analysis:
    - Curve: variance over draws of bn_1(q)/w_1 vs quota with CI
    - Boxplot: distribution of variance estimates over repeats
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha_{str(alpha).replace('.', 'p')}_theta_{str(theta).replace('.', 'p')}"
    curve_path = os.path.join(out_dir, f"{tag}_first_agent_variance_curve.pdf")
    boxplot_path = os.path.join(out_dir, f"{tag}_first_agent_variance_boxplot.pdf")

    Q = len(q_ratios)
    ratios_first_MQ = np.zeros((M, Q))
    for m in range(M):
        if M >= 5 and (m == 0 or (m + 1) % max(1, M // 5) == 0 or m == M - 1):
            print(f"    [run_first_agent curve] draw {m + 1}/{M}")
        W = draw_weights_gamma(n, alpha, theta)
        w_norm = W / np.sum(W)
        b_grid = mc_banzhaf_all_quota_vectorized(W, rp=rp, q_ratios=q_ratios, rng=rng)
        col_sums = b_grid.sum(axis=0, keepdims=True)
        bn_grid = np.divide(b_grid, col_sums, out=np.zeros_like(b_grid), where=col_sums != 0)
        ratios_first_MQ[m, :] = bn_grid[0, :] / w_norm[0]

    # Calculate Variance over draws
    means = ratios_first_MQ.mean(axis=0)
    centered = ratios_first_MQ - means
    s2 = (centered ** 2).sum(axis=0) / (M - 1) if M > 1 else np.zeros(Q)

    # Calculate Standard Error of s2
    if M > 3:
        m4 = (centered ** 4).sum(axis=0) / (M - 1)
        var_s2 = (m4 - ((M - 3) / (M - 1)) * (s2 ** 2)) / M
        se_s2 = np.sqrt(np.maximum(var_s2, 0))
    elif M > 1:
        se_s2 = np.sqrt(2 * (s2 ** 2) / (M - 1))
    else:
        se_s2 = np.zeros(Q)

    # FIX: Clip lower bound
    lo = np.maximum(0, s2 - 1.96 * se_s2)
    hi = s2 + 1.96 * se_s2

    # Plot Variance Curve
    plt.figure()
    plt.plot(q_ratios, s2, color='purple', label='Var over draws (Agent 1)')
    plt.fill_between(q_ratios, lo, hi, color='purple', alpha=0.2)
    plt.xlabel('Quota')
    plt.xlim(float(q_ratios.min()), float(q_ratios.max()))
    plt.ylabel('Variance of Ratio (Agent 1)')
    plt.title(f"Stability Analysis: Agent 1 Ratio Variance\nGamma(α={alpha}, θ={theta})")
    plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()

    # Boxplots for Representative Quotas
    if repr_qs is None:
        repr_qs = [0.10, 0.20, 0.40, 0.50, 0.60]
    q_indices, q_labels = [], []
    for q in repr_qs:
        idx = int(np.argmin(np.abs(q_ratios - q)))
        if (len(q_indices) == 0) or (idx != q_indices[-1]):
            q_indices.append(idx)
            q_labels.append(f"q={q_ratios[idx]:.2f}")

    var_samples_list = []
    for q_idx in q_indices:
        var_samples = np.zeros(repeats)
        for r in range(repeats):
            if repeats >= 5 and (r == 0 or (r + 1) % max(1, repeats // 5) == 0 or r == repeats - 1):
                print(f"    [run_first_agent repeats q={q_ratios[q_idx]:.2f}] repeat {r + 1}/{repeats}")
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

    plt.figure(figsize=(max(6, 1.8 * len(q_indices) + 2), 5))
    plt.boxplot(var_samples_list, widths=0.5, positions=np.arange(1, len(q_indices) + 1), vert=True,
                patch_artist=True, boxprops=dict(facecolor='lightsteelblue', alpha=0.7))
    plt.ylabel('Variance of Ratio (Agent 1)')
    plt.xticks(np.arange(1, len(q_indices) + 1), q_labels)
    plt.title(f"Variance Stability Check: Agent 1\n(Across {repeats} experiment repeats)")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(boxplot_path)
    plt.close()

    # Weight Distribution Boxplot (Single Agent)
    w1_values = []
    for m in range(M):
        W = draw_weights_gamma(n, alpha, theta)
        w_norm = W / np.sum(W)
        w1_values.append(w_norm[0])

    plt.figure(figsize=(6.5, 5.2))
    plt.boxplot([w1_values], widths=0.5, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#c2e0ff', alpha=0.8))
    plt.xticks([1], ["Agent 1 Normalized Weight"], rotation=0, ha='center')
    plt.ylabel('Weight (Proportion of Total)')
    plt.title(f"Agent 1 Weight Distribution (Gamma α={alpha}, θ={theta})")
    plt.grid(True, axis='y', alpha=0.3)
    try:
        plt.gcf().set_constrained_layout(True)
    except Exception:
        plt.tight_layout(rect=[0.06, 0.06, 0.98, 0.95])
    extra_path = os.path.join(out_dir, f"{tag}_first_agent_w1norm_boxplot.pdf")
    plt.savefig(extra_path)
    plt.close()


# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    # Base configuration
    default_n = 100
    M = 20
    rp = 10000

    # Output Directories
    plots_dir = os.path.join("plots", "curves_default")
    plots_weights_var_root = os.path.join("plots", "weights_var")
    plots_bars_root = os.path.join("plots", "bars")
    plots_large_n_root = os.path.join("plots", "large_n")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(plots_weights_var_root, exist_ok=True)
    os.makedirs(plots_bars_root, exist_ok=True)
    os.makedirs(plots_large_n_root, exist_ok=True)

    # Quota sets
    quota_sets = {
        'full_0_1_101': np.linspace(0, 1, 101),
        'focus_0p05_0p25_41': np.linspace(0.05, 0.25, 41),
        'focus_0p4_0p6_41': np.linspace(0.4, 0.6, 41),
    }

    # Larger-n configurations
    large_n_configs = [
        {'name': 'n500_full_0_1_21', 'n': 500, 'M': 20, 'rp': 6000, 'q_ratios': np.linspace(0, 1, 21)},
        {'name': 'n500_focus_0p05_0p25_21', 'n': 500, 'M': 20, 'rp': 6000, 'q_ratios': np.linspace(0.05, 0.25, 21)},
        {'name': 'n500_focus_0p4_0p6_31', 'n': 500, 'M': 20, 'rp': 8000, 'q_ratios': np.linspace(0.4, 0.6, 31)},
        {'name': 'n1000_full_0_1_21', 'n': 1000, 'M': 20, 'rp': 6000, 'q_ratios': np.linspace(0, 1, 21)},
        {'name': 'n1000_focus_0p05_0p25_21', 'n': 1000, 'M': 20, 'rp': 6000, 'q_ratios': np.linspace(0.05, 0.25, 21)},
        {'name': 'n1000_focus_0p4_0p6_31', 'n': 1000, 'M': 20, 'rp': 8000, 'q_ratios': np.linspace(0.4, 0.6, 31)},
    ]

    # Parameters: (alpha, theta)
    param_settings = [
        (0.24, 330235.0),  # Real data fit
        (0.5, 0.5), (0.5, 1.0), (0.5, 2.0),
        (1.0, 0.5), (1.0, 1.0)
    ]

    # First-agent knobs
    REPR_QS = [0.10, 0.20, 0.50, 0.60]
    REPEATS = 20

    master_rng = np.random.default_rng(42)

    # Progress tracking
    num_params = len(param_settings)
    jobs_per_param = 1 + 6 + 1 + 3 * len(large_n_configs)
    total_jobs = num_params * jobs_per_param
    job_idx = 0

    print(f"Starting Simulation with {total_jobs} total tasks...")

    for (idx_param, (alpha, theta)) in enumerate(param_settings, start=1):
        print(f"=== Param Set {idx_param}/{num_params}: alpha={alpha}, theta={theta} ===")
        tag_preview = f"a={alpha} t={theta}"
        rng = np.random.default_rng(master_rng.integers(0, 2 ** 63 - 1))

        # 1) Combined curve (Default n)
        job_idx += 1;
        print(f"[{job_idx}/{total_jobs}] Combined Curve (n={default_n})")
        run_combined_curve(alpha, theta, default_n, M, rp, plots_dir, rng, quota_sets['full_0_1_101'])

        # 2) Extended variants (Weights Var & Bars)
        for qname, q_ratios in quota_sets.items():
            out_weights_var = os.path.join(plots_weights_var_root, qname)
            out_bars = os.path.join(plots_bars_root, qname)
            os.makedirs(out_weights_var, exist_ok=True)
            os.makedirs(out_bars, exist_ok=True)

            job_idx += 1;
            print(f"[{job_idx}/{total_jobs}] Weights Var Plot (n={default_n}, {qname})")
            run_curve_with_wvar(alpha, theta, default_n, M, rp, out_weights_var, rng, q_ratios)

            job_idx += 1;
            print(f"[{job_idx}/{total_jobs}] Bar Summary (n={default_n}, {qname})")
            run_bars(alpha, theta, default_n, M, rp, out_bars, rng, q_ratios)

        # 3) Large-n variants
        for cfg in large_n_configs:
            out_dir = os.path.join(plots_large_n_root, cfg['name'])
            os.makedirs(out_dir, exist_ok=True)

            job_idx += 1;
            print(f"[{job_idx}/{total_jobs}] Large-n Combined ({cfg['name']})")
            run_combined_curve(alpha, theta, cfg['n'], cfg['M'], cfg['rp'], out_dir, rng, cfg['q_ratios'])

            job_idx += 1;
            print(f"[{job_idx}/{total_jobs}] Large-n Weights Var ({cfg['name']})")
            run_curve_with_wvar(alpha, theta, cfg['n'], cfg['M'], cfg['rp'], out_dir, rng, cfg['q_ratios'])

            job_idx += 1;
            print(f"[{job_idx}/{total_jobs}] Large-n Bars ({cfg['name']})")
            run_bars(alpha, theta, cfg['n'], cfg['M'], cfg['rp'], out_dir, rng, cfg['q_ratios'])

        # 4) First agent analysis
        job_idx += 1;
        print(f"[{job_idx}/{total_jobs}] First Agent Analysis (n={default_n})")
        run_first_agent(alpha, theta, default_n, M, rp, plots_dir, rng, quota_sets['full_0_1_101'], repr_qs=REPR_QS,
                        repeats=REPEATS)

    print("Done. All plots generated in 'plots/' directory.")