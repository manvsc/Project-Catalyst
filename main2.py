import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt

# Variance axis scaling
VARIANCE_YSCALE = "linear"  # "linear" or "log"
VARIANCE_EPS = 1e-12
VARIANCE_PAD_FRAC = 0.05

# --- CORE UTILITIES ---

def draw_weights_gamma(n, alpha, theta, rng):
    """Sample a length-n vector of agent weights from Gamma(α=alpha, θ=theta) using rng."""
    return rng.gamma(shape=alpha, scale=theta, size=n)

def make_rng(master_seed, tag):
    """Deterministic RNG from a master seed and a string tag."""
    digest = hashlib.sha256(f"{master_seed}|{tag}".encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    return np.random.default_rng(seed_int)


def apply_variance_scale(ax, scale=None):
    """Apply variance axis scaling and clamp to non-negative."""
    scale = VARIANCE_YSCALE if scale is None else scale
    if scale == "log":
        ax.set_yscale("log")
        y0, y1 = ax.get_ylim()
        y1 = max(y1, VARIANCE_EPS * 10)
        ax.set_ylim(bottom=VARIANCE_EPS, top=y1 * (1 + VARIANCE_PAD_FRAC))
    else:
        y0, y1 = ax.get_ylim()
        y1 = max(y1, 0.0)
        pad = VARIANCE_PAD_FRAC * (y1 if y1 > 0 else 1.0)
        ax.set_ylim(bottom=-pad, top=y1 + pad)
        ticks = [t for t in ax.get_yticks() if t >= 0]
        ax.set_yticks(ticks)


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
    include_sum = others_sum + W  # (rp,n), reused across quotas
    for j, q in enumerate(quotas):
        # Pivot condition: Losing without me (< q) AND Winning with me (>= q)
        pivots = (others_sum < q) & (include_sum >= q)
        b[:, j] = pivots.mean(axis=0)
    return b


# --- EXACT BANZHAF (FIRST AGENT ONLY) ---

def exact_banzhaf_first_agent_subset(W, q):
    """Exact pivot probability for agent 1 via subset enumeration (float weights)."""
    w1 = W[0]
    others = W[1:]
    sums = np.array([0.0])
    for w in others:
        sums = np.concatenate([sums, sums + w])
    pivots = (sums < q) & (sums + w1 >= q)
    return pivots.mean()


def exact_banzhaf_first_agent_dp_int(weights_int, quota_int):
    """Exact pivot probability for agent 1 using DP (integer weights)."""
    if quota_int <= 0:
        return 0.0
    total = int(np.sum(weights_int))
    if quota_int > total:
        return 0.0
    max_order = total
    polynomial = [1] + [0] * max_order
    current_order = 0
    aux_polynomial = polynomial[:]
    for w in weights_int:
        current_order += w
        offset = [0] * w + polynomial
        for j in range(current_order + 1):
            aux_polynomial[j] = polynomial[j] + offset[j]
        polynomial = aux_polynomial[:]

    w1 = int(weights_int[0])
    swings = [0] * quota_int
    for j in range(quota_int):
        if j < w1:
            swings[j] = polynomial[j]
        else:
            swings[j] = polynomial[j] - swings[j - w1]
    b1 = 0
    for k in range(w1):
        idx = quota_int - 1 - k
        if idx >= 0:
            b1 += swings[idx]
    total_coalitions = 2 ** (len(weights_int) - 1)
    return b1 / float(total_coalitions)


# --- SAMPLING HELPERS ---

def sample_intergroup_and_first_agent(alpha, theta, n, M, rp, q_ratios, rng):
    """Sample intergroup variance (across players) and first-agent ratios over M draws."""
    Q = len(q_ratios)
    intergroup_vars = np.zeros((M, Q))
    first_agent_ratios = np.zeros((M, Q))
    ddof_players = 1 if n > 1 else 0

    for m in range(M):
        if M >= 5 and (m == 0 or (m + 1) % max(1, M // 5) == 0 or m == M - 1):
            print(f"    [sample] draw {m + 1}/{M}")
        W = draw_weights_gamma(n, alpha, theta, rng)
        w_norm = W / np.sum(W)
        b_grid = mc_banzhaf_all_quota_vectorized(W, rp=rp, q_ratios=q_ratios, rng=rng)
        col_sums = b_grid.sum(axis=0, keepdims=True)
        bn_grid = np.divide(b_grid, col_sums, out=np.zeros_like(b_grid), where=col_sums != 0)
        ratios = bn_grid / w_norm[:, None]
        intergroup_vars[m, :] = np.maximum(ratios.var(axis=0, ddof=ddof_players), 0.0)
        first_agent_ratios[m, :] = ratios[0, :]

    return intergroup_vars, first_agent_ratios


def sample_mean_and_intergroup(alpha, theta, n, M, rp, q_ratios, rng):
    """Sample mean ratio (across players) and intergroup variance over M draws."""
    Q = len(q_ratios)
    mean_ratios = np.zeros((M, Q))
    intergroup_vars = np.zeros((M, Q))
    ddof_players = 1 if n > 1 else 0

    for m in range(M):
        if M >= 5 and (m == 0 or (m + 1) % max(1, M // 5) == 0 or m == M - 1):
            print(f"    [sample mean/var] draw {m + 1}/{M}")
        W = draw_weights_gamma(n, alpha, theta, rng)
        w_norm = W / np.sum(W)
        b_grid = mc_banzhaf_all_quota_vectorized(W, rp=rp, q_ratios=q_ratios, rng=rng)
        col_sums = b_grid.sum(axis=0, keepdims=True)
        bn_grid = np.divide(b_grid, col_sums, out=np.zeros_like(b_grid), where=col_sums != 0)
        ratios = bn_grid / w_norm[:, None]
        mean_ratios[m, :] = ratios.mean(axis=0)
        intergroup_vars[m, :] = np.maximum(ratios.var(axis=0, ddof=ddof_players), 0.0)

    return mean_ratios, intergroup_vars


def summarize_mean_ci(samples_MQ):
    """Return mean and 95% CI of samples (shape M x Q)."""
    M = samples_MQ.shape[0]
    mean = samples_MQ.mean(axis=0)
    se = samples_MQ.std(axis=0, ddof=1) / np.sqrt(M) if M > 1 else np.zeros_like(mean)
    lo = np.maximum(0, mean - 1.96 * se)
    hi = mean + 1.96 * se
    return mean, lo, hi


def summarize_variance_ci(samples_MQ):
    """Return variance across draws and an approximate 95% CI (normal approx)."""
    M = samples_MQ.shape[0]
    if M > 1:
        s2 = samples_MQ.var(axis=0, ddof=1)
        se_s2 = s2 * np.sqrt(2.0 / (M - 1))
    else:
        s2 = np.zeros(samples_MQ.shape[1])
        se_s2 = np.zeros_like(s2)
    lo = np.maximum(0, s2 - 1.96 * se_s2)
    hi = s2 + 1.96 * se_s2
    return s2, lo, hi


def format_param_label(alpha, theta, fitted_alpha, fitted_theta):
    """Short legend labels around fitted parameters."""
    tol = 1e-12
    if np.isclose(alpha, 0.5, atol=tol, rtol=0) and np.isclose(theta, 0.5, atol=tol, rtol=0):
        return r"$\alpha=0.5,\ \theta=0.5$"
    if np.isclose(alpha, fitted_alpha, atol=tol, rtol=0) and np.isclose(theta, fitted_theta, atol=tol, rtol=0):
        return r"$\alpha=\alpha_{\mathrm{fit}},\ \theta=\theta_{\mathrm{fit}}$"
    if np.isclose(alpha, fitted_alpha, atol=tol, rtol=0) and np.isclose(theta, fitted_theta * 0.5, atol=tol, rtol=0):
        return r"$\alpha=\alpha_{\mathrm{fit}},\ \theta=0.5\,\theta_{\mathrm{fit}}$"
    if np.isclose(alpha, fitted_alpha, atol=tol, rtol=0) and np.isclose(theta, fitted_theta * 2.0, atol=tol, rtol=0):
        return r"$\alpha=\alpha_{\mathrm{fit}},\ \theta=2\,\theta_{\mathrm{fit}}$"
    if np.isclose(theta, fitted_theta, atol=tol, rtol=0) and np.isclose(alpha, fitted_alpha * 0.5, atol=tol, rtol=0):
        return r"$\alpha=0.5\,\alpha_{\mathrm{fit}},\ \theta=\theta_{\mathrm{fit}}$"
    if np.isclose(theta, fitted_theta, atol=tol, rtol=0) and np.isclose(alpha, fitted_alpha * 2.0, atol=tol, rtol=0):
        return r"$\alpha=2\,\alpha_{\mathrm{fit}},\ \theta=\theta_{\mathrm{fit}}$"
    return rf"$\alpha={alpha:g},\ \theta={theta:g}$"


# --- PLOTTING FUNCTIONS ---


def run_intergroup_variance_curve(alpha, theta, n, M, rp, out_dir, rng, q_ratios):
    """Intergroup variance (across players) vs quota with 95% CI."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha_{str(alpha).replace('.', 'p')}_theta_{str(theta).replace('.', 'p')}"
    out_path = os.path.join(out_dir, f"{tag}_intergroup_variance_curve_n{n}.pdf")

    intergroup_vars, _ = sample_intergroup_and_first_agent(alpha, theta, n, M, rp, q_ratios, rng)
    mean_var, lo, hi = summarize_mean_ci(intergroup_vars)

    plt.figure(figsize=(8, 5.5))
    plt.plot(q_ratios, mean_var, color='darkred', label='Mean Intergroup Variance')
    plt.fill_between(q_ratios, lo, hi, color='lightcoral', alpha=0.3)
    plt.xlabel('Quota')
    plt.ylabel('Intergroup Variance of Ratio')
    plt.title(f"Intergroup Variance vs Quota (n={n})\nGamma(α={alpha}, θ={theta})")
    plt.xlim(float(q_ratios.min()), float(q_ratios.max()))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_first_agent_variance_multi_n(alpha, theta, n_list, M, rp, out_dir, master_seed, q_ratios,
                                     use_exact=False, exact_max_n=24, exact_method="subset",
                                     dp_weight_scale=10000, dp_max_total=200000,
                                     dp_mode="normalized", dp_allow_rescale=True):
    """First-agent mean/variance curves for multiple user counts (one figure)."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha_{str(alpha).replace('.', 'p')}_theta_{str(theta).replace('.', 'p')}"
    out_path = os.path.join(out_dir, f"{tag}_first_agent_mean_and_variance_multi_n.pdf")

    base_tag = f"first_agent_var_multi_n|alpha={alpha}|theta={theta}"
    fig, (ax_mean, ax_var) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for n in n_list:
        rng = make_rng(master_seed, base_tag)
        use_exact_local = use_exact
        if use_exact and exact_method == "subset" and (n - 1) > exact_max_n:
            use_exact_local = False
            print(f"    [first agent] n={n} too large for subset exact (max {exact_max_n + 1}); using MC")
        first_agent_ratios = np.zeros((M, len(q_ratios)))
        for m in range(M):
            if M >= 5 and (m == 0 or (m + 1) % max(1, M // 5) == 0 or m == M - 1):
                print(f"    [first agent] n={n} draw {m + 1}/{M}")
            W = draw_weights_gamma(n, alpha, theta, rng)
            w_norm = W / np.sum(W)
            total = np.sum(W)
            quotas = q_ratios * total

            b_grid = mc_banzhaf_all_quota_vectorized(W, rp=rp, q_ratios=q_ratios, rng=rng)
            col_sums = b_grid.sum(axis=0)

            if use_exact_local:
                if exact_method == "dp_int":
                    if dp_mode == "raw":
                        # Truncate raw weights after decimal (e.g., 521467.617 -> 521467)
                        weights_base = np.maximum(1, np.floor(W)).astype(int)
                    else:
                        # Truncate normalized weights after scaling
                        weights_base = np.maximum(1, np.floor(w_norm * dp_weight_scale)).astype(int)

                    total_int = int(weights_base.sum())
                    if total_int > dp_max_total:
                        if dp_allow_rescale:
                            factor = int(np.ceil(total_int / dp_max_total))
                            weights_int = np.maximum(1, (weights_base // factor)).astype(int)
                            total_int = int(weights_int.sum())
                            print(f"    [first agent] total_int too large; rescaled by {factor} to {total_int}")
                        else:
                            print(f"    [first agent] total_int={total_int} too large; using MC")
                            bn1_norm = np.divide(b_grid[0, :], col_sums, out=np.zeros_like(col_sums), where=col_sums != 0)
                            first_agent_ratios[m, :] = bn1_norm / w_norm[0]
                            continue
                    else:
                        weights_int = weights_base

                    quotas_int = np.floor(q_ratios * total_int).astype(int)
                    b1_exact = np.array([exact_banzhaf_first_agent_dp_int(weights_int, q) for q in quotas_int])
                    col_sums_adj = col_sums - b_grid[0, :] + b1_exact
                    bn1_norm = np.divide(b1_exact, col_sums_adj, out=np.zeros_like(b1_exact), where=col_sums_adj != 0)
                else:
                    b1_exact = np.array([exact_banzhaf_first_agent_subset(W, q) for q in quotas])
                    col_sums_adj = col_sums - b_grid[0, :] + b1_exact
                    bn1_norm = np.divide(b1_exact, col_sums_adj, out=np.zeros_like(b1_exact), where=col_sums_adj != 0)
            else:
                bn1_norm = np.divide(b_grid[0, :], col_sums, out=np.zeros_like(col_sums), where=col_sums != 0)

            first_agent_ratios[m, :] = bn1_norm / w_norm[0]

        mean_ratio, _, _ = summarize_mean_ci(first_agent_ratios)
        s2, _, _ = summarize_variance_ci(first_agent_ratios)
        s2 = np.maximum(s2, VARIANCE_EPS)
        ax_mean.plot(q_ratios, mean_ratio, label=f"n={n}")
        ax_var.plot(q_ratios, s2, label=f"n={n}")

    ax_mean.set_ylabel("Mean Power/Stake Ratio (Agent 1)")
    ax_mean.set_title(f"Agent 1 Mean Ratio vs Quota\nGamma(α={alpha}, θ={theta})")
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend()

    ax_var.set_xlabel('Quota')
    ax_var.set_ylabel('Variance of Ratio (Agent 1)')
    ax_var.set_title("Agent 1 Ratio Variance vs Quota")
    ax_var.set_xlim(float(q_ratios.min()), float(q_ratios.max()))
    apply_variance_scale(ax_var, scale="log")
    ax_var.grid(True, alpha=0.3)
    ax_var.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_intergroup_variance_multi_n(alpha, theta, n_list, M, rp, out_dir, master_seed, q_ratios):
    """Intergroup mean/variance curves for multiple user counts (one figure)."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha_{str(alpha).replace('.', 'p')}_theta_{str(theta).replace('.', 'p')}"
    out_path = os.path.join(out_dir, f"{tag}_mean_and_variance_multi_n.pdf")

    base_tag = f"multi_n|alpha={alpha}|theta={theta}"
    fig, (ax_mean, ax_var) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for n in n_list:
        rng = make_rng(master_seed, base_tag)
        mean_ratios, intergroup_vars = sample_mean_and_intergroup(alpha, theta, n, M, rp, q_ratios, rng)
        mean_of_means, _, _ = summarize_mean_ci(mean_ratios)
        mean_of_vars, _, _ = summarize_mean_ci(intergroup_vars)
        mean_of_vars = np.maximum(mean_of_vars, VARIANCE_EPS)
        ax_mean.plot(q_ratios, mean_of_means, label=f"n={n}")
        ax_var.plot(q_ratios, mean_of_vars, label=f"n={n}")

    ax_mean.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax_mean.set_ylabel("Mean Power/Stake Ratio")
    ax_mean.set_title(f"Mean Ratio vs Quota\nGamma(α={alpha}, θ={theta})")
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend()

    ax_var.set_xlabel('Quota')
    ax_var.set_ylabel('Intergroup Variance of Ratio')
    ax_var.set_title("Intergroup Variance vs Quota")
    ax_var.set_xlim(float(q_ratios.min()), float(q_ratios.max()))
    apply_variance_scale(ax_var, scale="log")
    ax_var.grid(True, alpha=0.3)
    ax_var.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_intergroup_variance_boxplot_fixed_quota(alpha, theta, n_list, M, rp, out_dir, master_seed, q_ratio_fixed):
    """Boxplots of intergroup mean/variance across draws for different n at a fixed quota."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha_{str(alpha).replace('.', 'p')}_theta_{str(theta).replace('.', 'p')}"
    out_path = os.path.join(out_dir, f"{tag}_intergroup_mean_and_variance_boxplot_q{q_ratio_fixed:.3f}.pdf")

    q_ratios = np.array([q_ratio_fixed])
    base_tag = f"boxplot_fixed_q|alpha={alpha}|theta={theta}|q={q_ratio_fixed}"
    labels = []
    mean_data = []
    var_data = []
    for n in n_list:
        rng = make_rng(master_seed, base_tag)
        mean_ratios, intergroup_vars = sample_mean_and_intergroup(alpha, theta, n, M, rp, q_ratios, rng)
        mean_data.append(mean_ratios[:, 0])
        var_data.append(np.maximum(intergroup_vars[:, 0], VARIANCE_EPS))
        labels.append(f"n={n}")

    fig, (ax_mean, ax_var) = plt.subplots(2, 1, figsize=(max(7, 1.5 * len(n_list) + 3), 8), sharex=True)
    ax_mean.boxplot(mean_data, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightsteelblue', alpha=0.7))
    ax_mean.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax_mean.set_ylabel('Mean Power/Stake Ratio')
    ax_mean.set_title(f"Mean Ratio at Fixed Quota q={q_ratio_fixed:.3f}")
    ax_mean.grid(True, axis='y', alpha=0.3)

    ax_var.boxplot(var_data, widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor='lightcoral', alpha=0.7))
    ax_var.set_xticks(np.arange(1, len(labels) + 1))
    ax_var.set_xticklabels(labels)
    ax_var.set_ylabel('Intergroup Variance of Ratio')
    ax_var.set_title("Intergroup Variance at Fixed Quota")
    apply_variance_scale(ax_var, scale="log")
    ax_var.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_intergroup_variance_multi_params(param_settings, n, M, rp, out_dir, master_seed, q_ratios, fitted_alpha, fitted_theta):
    """Intergroup mean/variance curves for multiple gamma parameter sets."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mean_and_variance_multi_params_n{n}.pdf")

    base_tag = f"multi_params|n={n}"
    fig, (ax_mean, ax_var) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for (alpha, theta) in param_settings:
        rng = make_rng(master_seed, base_tag)
        mean_ratios, intergroup_vars = sample_mean_and_intergroup(alpha, theta, n, M, rp, q_ratios, rng)
        mean_of_means, _, _ = summarize_mean_ci(mean_ratios)
        mean_of_vars, _, _ = summarize_mean_ci(intergroup_vars)
        label = format_param_label(alpha, theta, fitted_alpha, fitted_theta)
        ax_mean.plot(q_ratios, mean_of_means, label=label)
        ax_var.plot(q_ratios, mean_of_vars, label=label)

    ax_mean.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax_mean.set_ylabel("Mean Power/Stake Ratio")
    ax_mean.set_title(f"Mean Ratio vs Quota (n={n})\nMultiple Gamma Parameters")
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend(fontsize=8)

    ax_var.set_xlabel('Quota')
    ax_var.set_ylabel('Intergroup Variance of Ratio')
    ax_var.set_title("Intergroup Variance vs Quota")
    ax_var.set_xlim(float(q_ratios.min()), float(q_ratios.max()))
    apply_variance_scale(ax_var, scale="log")
    ax_var.grid(True, alpha=0.3)
    ax_var.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    # Base configuration
    default_n = 50
    M = 20
    rp = 15000

    # Output Directories
    plots_intergroup_root = os.path.join("plots2", "intergroup_variance_exact_first_agent")

    os.makedirs(plots_intergroup_root, exist_ok=True)

    # Quota sets
    quota_sets = {
        'full_0_1_101': np.linspace(0, 1, 101),
        'focus_0p05_0p25_41': np.linspace(0.05, 0.25, 41),
        'focus_0p4_0p6_41': np.linspace(0.4, 0.6, 41),
    }

    # Parameters: (alpha, theta)
    fitted_alpha = 0.273568
    fitted_theta = 1301506.236646
    theta_lo = fitted_theta * 0.5
    theta_hi = fitted_theta * 2.0
    alpha_lo = fitted_alpha * 0.5
    alpha_hi = fitted_alpha * 2.0
    param_settings = [
        (0.5, 0.5),
        (fitted_alpha, fitted_theta),  # data2.csv fit (fitting2.py)
        #(fitted_alpha, theta_lo),
        #(fitted_alpha, theta_hi),
        (alpha_lo, fitted_theta),
        (alpha_hi, fitted_theta),
    ]

    # Intergroup variance configs
    intergroup_n_list = [20, 40, 60, 80]
    intergroup_q_ratios = quota_sets['full_0_1_101']
    intergroup_q_fixed = 0.07

    # Exact Banzhaf (first agent only)
    USE_EXACT_FIRST_AGENT = True
    EXACT_FIRST_AGENT_MAX_N = 80
    EXACT_FIRST_AGENT_METHOD = "dp_int"  # "subset" (float exact) or "dp_int" (integer DP)
    EXACT_DP_WEIGHT_SCALE = 10000
    EXACT_DP_MAX_TOTAL = 200000
    EXACT_DP_MODE = "normalized"  # "normalized" or "raw"
    EXACT_DP_ALLOW_RESCALE = True

    # Reproducible master seed (all randomness derives from this)
    master_seed = 42

    # Progress tracking
    total_jobs = 4
    job_idx = 0

    print(f"Starting Simulation with {total_jobs} total tasks...")

    # Intergroup variance focused outputs (fitted params)
    alpha, theta = fitted_alpha, fitted_theta

    job_idx += 1
    print(f"[{job_idx}/{total_jobs}] First Agent Variance Multi-n")
    run_first_agent_variance_multi_n(
        alpha, theta, intergroup_n_list, M, rp, plots_intergroup_root, master_seed, intergroup_q_ratios,
        use_exact=USE_EXACT_FIRST_AGENT, exact_max_n=EXACT_FIRST_AGENT_MAX_N,
        exact_method=EXACT_FIRST_AGENT_METHOD, dp_weight_scale=EXACT_DP_WEIGHT_SCALE,
        dp_max_total=EXACT_DP_MAX_TOTAL, dp_mode=EXACT_DP_MODE, dp_allow_rescale=EXACT_DP_ALLOW_RESCALE
    )

    job_idx += 1
    print(f"[{job_idx}/{total_jobs}] Intergroup Mean+Variance Multi-n")
    run_intergroup_variance_multi_n(alpha, theta, intergroup_n_list, M, rp, plots_intergroup_root, master_seed, intergroup_q_ratios)

    job_idx += 1
    print(f"[{job_idx}/{total_jobs}] Intergroup Variance Boxplots (q={intergroup_q_fixed})")
    run_intergroup_variance_boxplot_fixed_quota(alpha, theta, intergroup_n_list, M, rp, plots_intergroup_root, master_seed, intergroup_q_fixed)

    job_idx += 1
    print(f"[{job_idx}/{total_jobs}] Intergroup Mean+Variance Multi-params (n={default_n})")
    run_intergroup_variance_multi_params(
        param_settings, default_n, M, rp, plots_intergroup_root, master_seed,
        intergroup_q_ratios, fitted_alpha, fitted_theta
    )

    print("Done. All plots generated in 'plots2/' directory.")
