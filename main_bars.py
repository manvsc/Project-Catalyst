import numpy as np
import os
import matplotlib.pyplot as plt

from main import draw_weights_gamma, mc_banzhaf_all_quota_vectorized


def run_setting(alpha, theta, n, M, rp, out_dir, rng, q_ratios):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha_{str(alpha).replace('.','p')}_theta_{str(theta).replace('.','p')}"
    bars_plot = os.path.join(out_dir, f"{tag}_combined_bars.pdf")

    Q = len(q_ratios)
    means_MQ = np.zeros((M, Q), dtype=float)
    vars_MQ = np.zeros((M, Q), dtype=float)

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

    # Top: mean of means as bars with CI
    ax_top.bar(x, mean_of_means, width=width, color='steelblue', alpha=0.8, label='Mean of means')
    ax_top.errorbar(x, mean_of_means, yerr=[mean_of_means - lo_mean, hi_mean - mean_of_means], fmt='none', ecolor='black', elinewidth=1, capsize=2)
    ax_top.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax_top.set_ylabel("Mean of (bn_i / w_i)")
    ax_top.grid(True, axis='y', alpha=0.3)

    # Bottom: mean of vars as bars with CI
    ax_bottom.bar(x, mean_of_vars, width=width, color='indianred', alpha=0.8, label='Mean of vars')
    ax_bottom.errorbar(x, mean_of_vars, yerr=[mean_of_vars - lo_var, hi_var - mean_of_vars], fmt='none', ecolor='black', elinewidth=1, capsize=2)
    ax_bottom.set_ylabel("Var of (bn_i / w_i)")
    ax_bottom.set_xlabel("quota index")
    ax_bottom.grid(True, axis='y', alpha=0.3)

    # Pretty x ticks every few quotas
    tick_idx = np.linspace(0, Q - 1, min(Q, 11), dtype=int)
    ax_bottom.set_xticks(tick_idx)
    ax_bottom.set_xticklabels([f"{q:.2f}" for q in q_ratios[tick_idx]])

    plt.suptitle(f"Gamma(α={alpha}, θ={theta}) — bar summary over quotas")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(bars_plot)
    plt.close()


if __name__ == "__main__":
    n = 100
    M = 100
    rp = 10000

    # Try different quota ranges / densities
    quota_sets = {
        'full_0_1_101': np.linspace(0, 1, 101),
        'focus_0p2_0p8_41': np.linspace(0.2, 0.8, 41),
        'dense_center_0p4_0p6_61': np.linspace(0.4, 0.6, 61),
    }

    param_settings = [
        (0.5, 0.5),
        (1.0, 1.0),
        (2.0, 1.0),
    ]

    master_rng = np.random.default_rng(123)

    for qname, q_ratios in quota_sets.items():
        out_dir = os.path.join("plots_ext", "bars", qname)
        for idx, (alpha, theta) in enumerate(param_settings, start=1):
            print(f"[bars:{qname}] {idx}/{len(param_settings)} alpha={alpha}, theta={theta}")
            rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
            run_setting(alpha, theta, n, M, rp, out_dir, rng, q_ratios)
    print("Done. Bar plots saved under plots_ext/bars/*")

