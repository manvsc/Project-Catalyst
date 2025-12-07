import numpy as np
import os
import matplotlib.pyplot as plt

from main import draw_weights_gamma, mc_banzhaf_all_quota_vectorized


def run_setting(alpha, theta, n, M, rp, out_dir, rng, q_ratios):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha_{str(alpha).replace('.','p')}_theta_{str(theta).replace('.','p')}"
    combined_plot = os.path.join(out_dir, f"{tag}_combined_curve.pdf")

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
    plt.title(f"Gamma(α={alpha}, θ={theta}) — large n={n}")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(combined_plot)
    plt.close()


if __name__ == "__main__":
    # Larger n variants; compensate by fewer quotas and/or smaller rp
    configs = [
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

    # Use same parameter set as in main.py
    param_settings = [
        (0.5, 0.5),
        # (0.5, 1.0), (0.5, 2.0),
        # (1.0, 0.5), (1.0, 1.0), (1.0, 2.0),
        # (2.0, 0.5), (2.0, 1.0), (2.0, 2.0),
        # (5.0, 1.0),
    ]

    master_rng = np.random.default_rng(7)

    for cfg in configs:
        out_dir = os.path.join("plots_ext", "large_n", cfg['name'])
        for idx, (alpha, theta) in enumerate(param_settings, start=1):
            print(f"[large_n:{cfg['name']}] {idx}/{len(param_settings)} alpha={alpha}, theta={theta}")
            rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
            run_setting(alpha, theta, cfg['n'], cfg['M'], cfg['rp'], out_dir, rng, cfg['q_ratios'])
    print("Done. Large-n plots saved under plots_ext/large_n/*")
