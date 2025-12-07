import os
import numpy as np
import matplotlib.pyplot as plt

# Core utilities
def draw_weights_gamma(n, alpha, theta):
    return np.random.gamma(shape=alpha, scale=theta, size=n)

def mc_banzhaf_all_quota_vectorized(W, rp, q_ratios, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = len(W)
    Q = len(q_ratios)
    total = np.sum(W)
    quotas = q_ratios * total

    T = rng.integers(0, 2, size=(rp, n), dtype=np.int8)
    W_mat = T * W
    sum_total = W_mat.sum(axis=1)
    others_sum = sum_total[:, None] - W_mat

    b = np.zeros((n, Q))
    for j, q in enumerate(quotas):
        include_sum = others_sum + W
        pivots = (others_sum < q) & (include_sum >= q)
        b[:, j] = pivots.mean(axis=0)
    return b

# Plot 1: Combined curve (mean/var of ratios) like main.py
def run_combined_curve(alpha, theta, n, M, rp, out_dir, rng, q_ratios):
    """Combined curve plot over quotas showing:
    - Left axis: mean of bn_i/w_i across players (with 95% CI across draws)
    - Right axis: variance of bn_i/w_i across players (with 95% CI across draws)
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha_{str(alpha).replace('.','p')}_theta_{str(theta).replace('.','p')}"
    out_path = os.path.join(out_dir, f"{tag}_combined_curve.pdf")

    Q = len(q_ratios)
    means_MQ = np.zeros((M, Q))
    vars_MQ = np.zeros((M, Q))

    for m in range(M):
        W = draw_weights_gamma(n, alpha, theta)
        w_norm = W / np.sum(W)
        b_grid = mc_banzhaf_all_quota_vectorized(W, rp=rp, q_ratios=q_ratios, rng=rng)
        col_sums = b_grid.sum(axis=0, keepdims=True)
        bn_grid = np.divide(b_grid, col_sums, out=np.zeros_like(b_grid), where=col_sums != 0)
        ratios = bn_grid / w_norm[:, None]
        means_MQ[m, :] = ratios.mean(axis=0)
        vars_MQ[m, :] = ratios.var(axis=0)

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
    ax1.plot(q_ratios, mean_of_means, color='blue', label='Mean of means')
    ax1.fill_between(q_ratios, lo_mean, hi_mean, color='blue', alpha=0.2)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax1.set_ylabel("Mean of (bn_i / w_i)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.plot(q_ratios, mean_of_vars, color='red', linestyle='--', label='Mean of vars')
    ax2.fill_between(q_ratios, lo_var, hi_var, color='red', alpha=0.2)
    ax2.set_ylabel("Variance of (bn_i / w_i)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax1.set_xlabel("quota")
    ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Gamma(α={alpha}, θ={theta}) — normalized Banzhaf ratios")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Plot 2: Weights variance panel like main_weights_var.py
def run_curve_with_wvar(alpha, theta, n, M, rp, out_dir, rng, q_ratios):
    """Two-panel figure:
    - Top: same combined curves as run_combined_curve
    - Bottom: distribution (mean+95% CI) of variance of normalized weights across draws
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha_{str(alpha).replace('.','p')}_theta_{str(theta).replace('.','p')}"
    out_path = os.path.join(out_dir, f"{tag}_combined_curve_with_wvar.pdf")

    Q = len(q_ratios)
    means_MQ = np.zeros((M, Q))
    vars_MQ = np.zeros((M, Q))
    wvar_M = np.zeros(M)

    for m in range(M):
        W = draw_weights_gamma(n, alpha, theta)
        w_norm = W / np.sum(W)
        wvar_M[m] = np.var(w_norm)
        b_grid = mc_banzhaf_all_quota_vectorized(W, rp=rp, q_ratios=q_ratios, rng=rng)
        col_sums = b_grid.sum(axis=0, keepdims=True)
        bn_grid = np.divide(b_grid, col_sums, out=np.zeros_like(b_grid), where=col_sums != 0)
        ratios = bn_grid / w_norm[:, None]
        means_MQ[m, :] = ratios.mean(axis=0)
        vars_MQ[m, :] = ratios.var(axis=0)

    mean_of_means = means_MQ.mean(axis=0)
    se_means = means_MQ.std(axis=0, ddof=1) / np.sqrt(M) if M > 1 else np.zeros(Q)
    lo_mean = mean_of_means - 1.96 * se_means
    hi_mean = mean_of_means + 1.96 * se_means
    mean_of_vars = vars_MQ.mean(axis=0)
    se_vars = vars_MQ.std(axis=0, ddof=1) / np.sqrt(M) if M > 1 else np.zeros(Q)
    lo_var = mean_of_vars - 1.96 * se_vars
    hi_var = mean_of_vars + 1.96 * se_vars
    wvar_mean = float(np.mean(wvar_M))
    wvar_se = float(np.std(wvar_M, ddof=1) / np.sqrt(M)) if M > 1 else 0.0
    wvar_ci = (wvar_mean - 1.96 * wvar_se, wvar_mean + 1.96 * wvar_se)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1b = ax1.twinx()
    ax1.plot(q_ratios, mean_of_means, color='blue')
    ax1.fill_between(q_ratios, lo_mean, hi_mean, color='blue', alpha=0.2)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax1.set_ylabel("Mean of (bn_i / w_i)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    ax1b.plot(q_ratios, mean_of_vars, color='red', linestyle='--')
    ax1b.fill_between(q_ratios, lo_var, hi_var, color='red', alpha=0.2)
    ax1b.set_ylabel("Variance of (bn_i / w_i)", color='red')
    ax1b.tick_params(axis='y', labelcolor='red')
    ax1.set_xlabel("quota")
    ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
    ax1.set_title(f"Gamma(α={alpha}, θ={theta}) — ratios and weight variance")
    ax2.bar([0], [wvar_mean], width=0.6, color='purple', alpha=0.8)
    ax2.errorbar([0], [wvar_mean], yerr=[[wvar_mean - wvar_ci[0]], [wvar_ci[1] - wvar_mean]], fmt='none', ecolor='black', capsize=3)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Var(w_norm)"])
    ax2.set_ylabel("Variance of normalized weights")
    ax2.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Plot 3: Bars summary like main_bars.py
def run_bars(alpha, theta, n, M, rp, out_dir, rng, q_ratios):
    """Bar summary of mean and variance of bn_i/w_i across quotas with 95% CI bars."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha_{str(alpha).replace('.','p')}_theta_{str(theta).replace('.','p')}"
    out_path = os.path.join(out_dir, f"{tag}_combined_bars.pdf")

    Q = len(q_ratios)
    means_MQ = np.zeros((M, Q))
    vars_MQ = np.zeros((M, Q))
    for m in range(M):
        W = draw_weights_gamma(n, alpha, theta)
        w_norm = W / np.sum(W)
        b_grid = mc_banzhaf_all_quota_vectorized(W, rp=rp, q_ratios=q_ratios, rng=rng)
        col_sums = b_grid.sum(axis=0, keepdims=True)
        bn_grid = np.divide(b_grid, col_sums, out=np.zeros_like(b_grid), where=col_sums != 0)
        ratios = bn_grid / w_norm[:, None]
        means_MQ[m, :] = ratios.mean(axis=0)
        vars_MQ[m, :] = ratios.var(axis=0)

    mean_of_means = means_MQ.mean(axis=0)
    se_means = means_MQ.std(axis=0, ddof=1) / np.sqrt(M) if M > 1 else np.zeros(Q)
    lo_mean = mean_of_means - 1.96 * se_means
    hi_mean = mean_of_means + 1.96 * se_means
    mean_of_vars = vars_MQ.mean(axis=0)
    se_vars = vars_MQ.std(axis=0, ddof=1) / np.sqrt(M) if M > 1 else np.zeros(Q)
    lo_var = mean_of_vars - 1.96 * se_vars
    hi_var = mean_of_vars + 1.96 * se_vars

    x = np.arange(Q)
    width = 0.6
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_top.bar(x, mean_of_means, width=width, color='steelblue', alpha=0.8)
    ax_top.errorbar(x, mean_of_means, yerr=[mean_of_means - lo_mean, hi_mean - mean_of_means], fmt='none', ecolor='black', elinewidth=1, capsize=2)
    ax_top.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax_top.set_ylabel("Mean of (bn_i / w_i)")
    ax_top.grid(True, axis='y', alpha=0.3)
    ax_bottom.bar(x, mean_of_vars, width=width, color='indianred', alpha=0.8)
    ax_bottom.errorbar(x, mean_of_vars, yerr=[mean_of_vars - lo_var, hi_var - mean_of_vars], fmt='none', ecolor='black', elinewidth=1, capsize=2)
    ax_bottom.set_ylabel("Var of (bn_i / w_i)")
    ax_bottom.set_xlabel("quota index")
    ax_bottom.grid(True, axis='y', alpha=0.3)
    tick_idx = np.linspace(0, Q - 1, min(Q, 11), dtype=int)
    ax_bottom.set_xticks(tick_idx)
    ax_bottom.set_xticklabels([f"{q:.2f}" for q in q_ratios[tick_idx]])
    plt.suptitle(f"Gamma(α={alpha}, θ={theta}) — bar summary over quotas")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path)
    plt.close()

# Plot 4: First-agent variance + multi-quota violin/box
def run_first_agent(alpha, theta, n, M, rp, out_dir, rng, q_ratios):
    """First-agent analysis:
    - Curve: variance over draws of bn_1(q)/w_1 vs quota with CI
    - Violin+box: distribution of variance estimates over 100 repeats at representative quotas
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha_{str(alpha).replace('.', 'p')}_theta_{str(theta).replace('.', 'p')}"
    curve_path = os.path.join(out_dir, f"{tag}_first_agent_variance_curve.pdf")
    boxplot_path = os.path.join(out_dir, f"{tag}_first_agent_variance_boxplot.pdf")

    Q = len(q_ratios)
    ratios_first_MQ = np.zeros((M, Q))
    for m in range(M):
        W = draw_weights_gamma(n, alpha, theta)
        w_norm = W / np.sum(W)
        b_grid = mc_banzhaf_all_quota_vectorized(W, rp=rp, q_ratios=q_ratios, rng=rng)
        col_sums = b_grid.sum(axis=0, keepdims=True)
        bn_grid = np.divide(b_grid, col_sums, out=np.zeros_like(b_grid), where=col_sums != 0)
        ratios_first_MQ[m, :] = bn_grid[0, :] / w_norm[0]

    means = ratios_first_MQ.mean(axis=0)
    centered = ratios_first_MQ - means
    s2 = (centered**2).sum(axis=0) / (M - 1) if M > 1 else np.zeros(Q)
    if M > 3:
        m4 = (centered**4).sum(axis=0) / (M - 1)
        var_s2 = (m4 - ((M - 3) / (M - 1)) * (s2**2)) / M
        se_s2 = np.sqrt(np.maximum(var_s2, 0))
    elif M > 1:
        se_s2 = np.sqrt(2 * (s2**2) / (M - 1))
    else:
        se_s2 = np.zeros(Q)
    lo, hi = s2 - 1.96 * se_s2, s2 + 1.96 * se_s2
    plt.figure()
    plt.plot(q_ratios, s2, color='purple', label='Var over draws (player 1)')
    plt.fill_between(q_ratios, lo, hi, color='purple', alpha=0.2)
    plt.xlabel('quota')
    plt.ylabel('Variance of bn_1/w_1 over draws')
    plt.title(f"Gamma(α={alpha}, θ={theta}) — First agent ratio variance")
    plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()

    repr_qs = [0.10, 0.20, 0.40, 0.50, 0.60]
    q_indices, q_labels = [], []
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
    # Base configuration
    default_n = 100
    M = 100
    rp = 10000
    plots_dir = "plots"
    plots_ext = "plots_ext"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(plots_ext, exist_ok=True)

    # Quota sets, uniform densities (no dense center)
    quota_sets = {
        'full_0_1_101': np.linspace(0, 1, 101),
        'focus_0p05_0p25_41': np.linspace(0.05, 0.25, 41),
        'focus_0p4_0p6_41': np.linspace(0.4, 0.6, 41),
    }

    # Larger-n configurations inspired by main_large_n.py
    large_n_configs = [
        {
            'name': 'n500_focus_0p4_0p6_31',
            'n': 500, 'M': 100, 'rp': 8000,
            'q_ratios': np.linspace(0.4, 0.6, 31)
        },
        {
            'name': 'n1000_focus_0p05_0p25_21',
            'n': 1000, 'M': 80, 'rp': 6000,
            'q_ratios': np.linspace(0.05, 0.25, 21)
        },
        {
            'name': 'n500_full_0_1_21',
            'n': 500, 'M': 100, 'rp': 6000,
            'q_ratios': np.linspace(0, 1, 21)
        },
    ]

    # Start with a single parameter to preview
    param_settings = [
        (0.5, 0.5),
        # (0.5, 1.0), (0.5, 2.0),
        # (1.0, 0.5), (1.0, 1.0), (1.0, 2.0),
        # (2.0, 0.5), (2.0, 1.0), (2.0, 2.0),
        # (5.0, 1.0),
    ]

    master_rng = np.random.default_rng(42)

    for (alpha, theta) in param_settings:
        rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
        # 1) Combined curve on full range at default n
        run_combined_curve(alpha, theta, default_n, M, rp, plots_dir, rng, quota_sets['full_0_1_101'])
        # 2) Extended variants per quota set at default n
        for qname, q_ratios in quota_sets.items():
            out_weights_var = os.path.join(plots_ext, 'weights_var', qname)
            out_bars = os.path.join(plots_ext, 'bars', qname)
            os.makedirs(out_weights_var, exist_ok=True)
            os.makedirs(out_bars, exist_ok=True)
            run_curve_with_wvar(alpha, theta, default_n, M, rp, out_weights_var, rng, q_ratios)
            run_bars(alpha, theta, default_n, M, rp, out_bars, rng, q_ratios)
        # 3) First agent variance (curve + multi-quota 100x repeats) at default n
        run_first_agent(alpha, theta, default_n, M, rp, plots_dir, rng, quota_sets['full_0_1_101'])

        # 4) Large-n variants
        for cfg in large_n_configs:
            out_dir = os.path.join(plots_ext, 'large_n', cfg['name'])
            os.makedirs(out_dir, exist_ok=True)
            # 4a) Combined curve
            run_combined_curve(alpha, theta, cfg['n'], cfg['M'], cfg['rp'], out_dir, rng, cfg['q_ratios'])
            # 4b) Weights variance panel (same quotas as cfg)
            run_curve_with_wvar(alpha, theta, cfg['n'], cfg['M'], cfg['rp'], out_dir, rng, cfg['q_ratios'])
            # 4c) Bars summary (same quotas as cfg)
            run_bars(alpha, theta, cfg['n'], cfg['M'], cfg['rp'], out_dir, rng, cfg['q_ratios'])

    print("Done. See plots/, plots_ext/weights_var, plots_ext/bars, and plots_ext/large_n.")
