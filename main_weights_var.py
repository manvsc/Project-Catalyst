import numpy as np
import os
import matplotlib.pyplot as plt

from main import draw_weights_gamma, mc_banzhaf_all_quota_vectorized


def run_setting(alpha, theta, n, M, rp, out_dir, rng, q_ratios):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha_{str(alpha).replace('.','p')}_theta_{str(theta).replace('.','p')}"
    combined_plot = os.path.join(out_dir, f"{tag}_combined_curve_with_wvar.pdf")

    Q = len(q_ratios)
    means_MQ = np.zeros((M, Q), dtype=float)
    vars_MQ = np.zeros((M, Q), dtype=float)
    wvar_M = np.zeros(M, dtype=float)

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

    # Weight variance summary across M (constant across quotas)
    wvar_mean = float(np.mean(wvar_M))
    wvar_se = float(np.std(wvar_M, ddof=1) / np.sqrt(M)) if M > 1 else 0.0
    wvar_ci = (wvar_mean - 1.96 * wvar_se, wvar_mean + 1.96 * wvar_se)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    # Top: same combined curves
    ax1b = ax1.twinx()
    ax1.plot(q_ratios, mean_of_means, color='blue', label='Mean of means')
    ax1.fill_between(q_ratios, lo_mean, hi_mean, color='blue', alpha=0.2)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax1.set_ylabel("Mean of (bn_i / w_i)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)

    ax1b.plot(q_ratios, mean_of_vars, color='red', linestyle='--', label='Mean of vars')
    ax1b.fill_between(q_ratios, lo_var, hi_var, color='red', alpha=0.2)
    ax1b.set_ylabel("Variance of (bn_i / w_i)", color='red')
    ax1b.tick_params(axis='y', labelcolor='red')
    ax1.set_xlabel("quota")
    ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
    ax1.set_title(f"Gamma(α={alpha}, θ={theta}) — ratios and weight variance")

    # Bottom: weight variance summary with CI as bar
    ax2.bar([0], [wvar_mean], width=0.6, color='purple', alpha=0.8)
    ax2.errorbar([0], [wvar_mean], yerr=[[wvar_mean - wvar_ci[0]], [wvar_ci[1] - wvar_mean]], fmt='none', ecolor='black', capsize=3)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Var(w_norm)"])
    ax2.set_ylabel("Variance of normalized weights")
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(combined_plot)
    plt.close()


if __name__ == "__main__":
    n = 100
    M = 100
    rp = 10000

    quota_sets = {
        'full_0_1_101': np.linspace(0, 1, 101),
        'focus_0p3_0p7_31': np.linspace(0.3, 0.7, 31),
        'center_dense_0p45_0p55_41': np.linspace(0.45, 0.55, 41),
    }

    param_settings = [
        (0.5, 0.5),
        (1.0, 1.0),
        (2.0, 1.0),
    ]

    master_rng = np.random.default_rng(99)

    for qname, q_ratios in quota_sets.items():
        out_dir = os.path.join("plots_ext", "weights_var", qname)
        for idx, (alpha, theta) in enumerate(param_settings, start=1):
            print(f"[weights_var:{qname}] {idx}/{len(param_settings)} alpha={alpha}, theta={theta}")
            rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
            run_setting(alpha, theta, n, M, rp, out_dir, rng, q_ratios)
    print("Done. Weights-variance plots saved under plots_ext/weights_var/*")

