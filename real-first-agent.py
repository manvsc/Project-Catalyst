import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt

# Variance axis scaling (force linear)
VARIANCE_YSCALE = "linear"
VARIANCE_PAD_FRAC = 0.05


def draw_weights_gamma(n, alpha, theta, rng):
    """Sample a length-n vector of agent weights from Gamma(alpha=alpha, theta=theta) using rng."""
    return rng.gamma(shape=alpha, scale=theta, size=n)


def make_rng(master_seed, tag):
    """Deterministic RNG from a master seed and a string tag."""
    digest = hashlib.sha256(f"{master_seed}|{tag}".encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    return np.random.default_rng(seed_int)


def apply_variance_scale(ax):
    """Apply linear variance axis scaling and clamp to non-negative ticks."""
    y0, y1 = ax.get_ylim()
    y1 = max(y1, 0.0)
    pad = VARIANCE_PAD_FRAC * (y1 if y1 > 0 else 1.0)
    ax.set_ylim(bottom=-pad, top=y1 + pad)
    ticks = [t for t in ax.get_yticks() if t >= 0]
    ax.set_yticks(ticks)


def exact_banzhaf_first_agent_dp_int_all_quotas(weights_int, quotas_int):
    """Exact unnormalized Banzhaf (pivot probability) for agent 1 across quotas.

    weights_int: integer weights for all agents (length n)
    quotas_int: integer quotas (array-like)

    Returns:
        b1: array of pivot probabilities for agent 1, one per quota.
    """
    weights_int = np.asarray(weights_int, dtype=int)
    quotas_int = np.asarray(quotas_int, dtype=int)

    n = len(weights_int)
    if n == 0:
        return np.zeros_like(quotas_int, dtype=float)

    w1 = int(weights_int[0])
    total = int(weights_int.sum())
    total_other = total - w1
    total_coalitions = 2 ** (n - 1)

    # DP counts for subset sums of the other agents
    poly_other = np.zeros(total_other + 1, dtype=np.int64)
    poly_other[0] = 1
    for w in weights_int[1:]:
        w = int(w)
        poly_other[w:] += poly_other[:-w]

    prefix = np.zeros(total_other + 2, dtype=np.int64)
    prefix[1:] = np.cumsum(poly_other, dtype=np.int64)

    b1 = np.zeros_like(quotas_int, dtype=float)
    for i, q in enumerate(quotas_int):
        if q <= 0 or q > total:
            b1[i] = 0.0
            continue
        lo = q - w1
        if lo < 0:
            lo = 0
        hi = q - 1
        if hi > total_other:
            hi = total_other
        if hi < lo:
            b1[i] = 0.0
            continue
        swings = prefix[hi + 1] - prefix[lo]
        b1[i] = swings / float(total_coalitions)

    return b1


def compute_first_agent_ratios_exact(alpha, theta, n, M, q_ratios, rng,
                                     dp_weight_scale=10000, dp_max_total=200000,
                                     dp_mode="normalized", dp_allow_rescale=True):
    """Compute unnormalized Banzhaf/weight ratios for agent 1 across draws and quotas."""
    Q = len(q_ratios)
    ratios = np.zeros((M, Q))

    for m in range(M):
        if M >= 5 and (m == 0 or (m + 1) % max(1, M // 5) == 0 or m == M - 1):
            print(f"    [exact] draw {m + 1}/{M}")

        W = draw_weights_gamma(n, alpha, theta, rng)
        w_norm = W / np.sum(W)

        if dp_mode == "raw":
            weights_base = np.maximum(1, np.floor(W)).astype(int)
        else:
            weights_base = np.maximum(1, np.floor(w_norm * dp_weight_scale)).astype(int)

        total_int = int(weights_base.sum())
        if total_int > dp_max_total:
            if not dp_allow_rescale:
                raise ValueError(f"total_int={total_int} exceeds dp_max_total={dp_max_total}")
            factor = int(np.ceil(total_int / dp_max_total))
            weights_int = np.maximum(1, (weights_base // factor)).astype(int)
            total_int = int(weights_int.sum())
            print(f"    [exact] total_int too large; rescaled by {factor} to {total_int}")
        else:
            weights_int = weights_base

        quotas_int = np.floor(q_ratios * total_int).astype(int)
        b1 = exact_banzhaf_first_agent_dp_int_all_quotas(weights_int, quotas_int)

        ratios[m, :] = b1 / w_norm[0]

    return ratios


def summarize_mean_and_variance(samples_MQ):
    """Return mean and sample variance across draws for each quota."""
    M = samples_MQ.shape[0]
    mean = samples_MQ.mean(axis=0)
    var = samples_MQ.var(axis=0, ddof=1) if M > 1 else np.zeros_like(mean)
    return mean, var


def run_real_first_agent(alpha, theta, n, M, q_ratios, out_dir, master_seed,
                         dp_weight_scale=10000, dp_max_total=200000,
                         dp_mode="normalized", dp_allow_rescale=True):
    """Run exact (unnormalized) Banzhaf for agent 1 and plot mean/variance curves."""
    os.makedirs(out_dir, exist_ok=True)

    rng = make_rng(master_seed, f"real_first_agent|alpha={alpha}|theta={theta}|n={n}")

    ratios = compute_first_agent_ratios_exact(
        alpha, theta, n, M, q_ratios, rng,
        dp_weight_scale=dp_weight_scale,
        dp_max_total=dp_max_total,
        dp_mode=dp_mode,
        dp_allow_rescale=dp_allow_rescale,
    )
    mean_ratio, var_ratio = summarize_mean_and_variance(ratios)

    tag = f"alpha_{str(alpha).replace('.', 'p')}_theta_{str(theta).replace('.', 'p')}_n{n}"
    out_path = os.path.join(out_dir, f"{tag}_real_first_agent_mean_variance.pdf")

    fig, (ax_mean, ax_var) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax_mean.plot(q_ratios, mean_ratio, color="navy")
    ax_mean.set_ylabel("Mean Ratio (Unnormalized Banzhaf / Normalized Weight)")
    ax_mean.set_title(f"Agent 1 Mean Ratio vs Quota\nGamma(alpha={alpha}, theta={theta})")
    ax_mean.grid(True, alpha=0.3)

    ax_var.plot(q_ratios, var_ratio, color="darkred")
    ax_var.set_xlabel("Quota")
    ax_var.set_ylabel("Variance Across Draws")
    ax_var.set_title("Agent 1 Ratio Variance vs Quota")
    ax_var.set_xlim(float(q_ratios.min()), float(q_ratios.max()))
    apply_variance_scale(ax_var)
    ax_var.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return out_path


def run_real_first_agent_multi_n(alpha, theta, n_list, M, q_ratios, out_dir, master_seed,
                                 dp_weight_scale=10000, dp_max_total=200000,
                                 dp_mode="normalized", dp_allow_rescale=True):
    """Run exact (unnormalized) Banzhaf for agent 1 across multiple n values."""
    os.makedirs(out_dir, exist_ok=True)

    tag = f"alpha_{str(alpha).replace('.', 'p')}_theta_{str(theta).replace('.', 'p')}"
    out_path = os.path.join(out_dir, f"{tag}_real_first_agent_mean_variance_multi_n.pdf")

    fig, (ax_mean, ax_var) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for n in n_list:
        rng = make_rng(master_seed, f"real_first_agent|alpha={alpha}|theta={theta}|n={n}")
        ratios = compute_first_agent_ratios_exact(
            alpha, theta, n, M, q_ratios, rng,
            dp_weight_scale=dp_weight_scale,
            dp_max_total=dp_max_total,
            dp_mode=dp_mode,
            dp_allow_rescale=dp_allow_rescale,
        )
        mean_ratio, var_ratio = summarize_mean_and_variance(ratios)

        ax_mean.plot(q_ratios, mean_ratio, label=f"n={n}")
        ax_var.plot(q_ratios, var_ratio, label=f"n={n}")

    ax_mean.set_ylabel("Mean Ratio (Unnormalized Banzhaf / Normalized Weight)")
    ax_mean.set_title(f"Agent 1 Mean Ratio vs Quota\nGamma(alpha={alpha}, theta={theta})")
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend()

    ax_var.set_xlabel("Quota")
    ax_var.set_ylabel("Variance Across Draws")
    ax_var.set_title("Agent 1 Ratio Variance vs Quota")
    ax_var.set_xlim(float(q_ratios.min()), float(q_ratios.max()))
    apply_variance_scale(ax_var)
    ax_var.grid(True, alpha=0.3)
    ax_var.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return out_path


if __name__ == "__main__":
    # Base configuration
    n_list = [30, 40, 60, 80]
    M = 20

    # Quota grid
    q_ratios = np.linspace(0, 1, 101)

    # Fitted Gamma parameters
    fitted_alpha = 1
    fitted_theta = 0.5

    # Exact DP settings
    DP_WEIGHT_SCALE = 10000
    DP_MAX_TOTAL = 200000
    DP_MODE = "normalized"  # "normalized" or "raw"
    DP_ALLOW_RESCALE = True

    # Reproducible master seed
    master_seed = 42

    out_dir = os.path.join("plots2", "real_first_agent")

    print("Running real (exact) first-agent Banzhaf (multi-n)...")
    out_path = run_real_first_agent_multi_n(
        fitted_alpha,
        fitted_theta,
        n_list,
        M,
        q_ratios,
        out_dir,
        master_seed,
        dp_weight_scale=DP_WEIGHT_SCALE,
        dp_max_total=DP_MAX_TOTAL,
        dp_mode=DP_MODE,
        dp_allow_rescale=DP_ALLOW_RESCALE,
    )
    print(f"Done. Plot saved to {out_path}")
