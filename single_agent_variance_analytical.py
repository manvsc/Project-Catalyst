import numpy as np
from math import comb
from scipy.integrate import quad
from scipy.special import betaln, betainc
import matplotlib.pyplot as plt

# ---------- Core building blocks ----------

def px1(c: float, n: int, alpha: float) -> float:
    """
    P_{X1}(c) = Gamma(n a) / (Gamma(a) Gamma((n-1)a)) * c^(a-1) * (1-c)^((n-1)a - 1)
              = BetaPDF(c; a, (n-1)a)
    computed stably via logs.
    """
    if c <= 0 or c >= 1:
        return 0
    a = alpha
    b = (n - 1) * alpha
    log_pdf = (a - 1) * np.log(c) + (b - 1) * np.log(1 - c) - betaln(a, b)
    return float(np.exp(log_pdf))


def Delta(a: float, k: int, n: int, alpha: float) -> float:
    """
    Δ(a; k, α) = P[ Beta(k, α) <= a ].
    Uses regularized incomplete beta: betainc(k, α, a).
    Clamps to [0,1] for numerical safety.
    """
    if a <= 0:
        return 0
    if a >= 1:
        return 1
    return float(betainc(k*alpha, (n-1-k)*alpha, a))


def S(c: float, theta: float, n: int, alpha: float) -> float:
    """
    S(c) = sum_{k=1}^{n-2} [ C(n-1,k) / 2^(n-1) ] * ( Δ(theta/(1-c); k, α) - Δ((theta-c)/(1-c); k, α) )
    """
    if n < 3:
        return 0

    denom = 1 - c
    # When c -> 1, denom -> 0; we won't call S(c) in the last integral region.
    t1 = theta / denom
    t2 = (theta - c) / denom

    w_denom = 2 ** (n - 1)
    acc = 0
    for k in range(1, n - 1):  # 1..n-2 inclusive via condition
        w = comb(n - 1, k) / w_denom
        acc += w * (Delta(t1, k, n, alpha) - Delta(t2, k, n, alpha))
    return acc

def S2(c: float, theta: float, n: int, alpha: float) -> float:
    """
    S2(c) = sum_{k=1}^{n-2} C(n-1,k)/2^(n-1) * ( 1 - Δ((theta-c)/(1-c);k,α) ) + 1/2^(n-1)
    """
    if n < 3:
        return 1.0 / (2 ** (n - 1))
    denom = 1.0 - c
    t2 = (theta - c) / denom

    w_denom = 2 ** (n - 1)
    acc = 0.0
    for k in range(1, n - 1):  # 1..n-2
        acc += (comb(n - 1, k) / w_denom) * (1.0 - Delta(t2, k, n, alpha))
    acc += 1.0 / w_denom
    return acc

def quad_zero_singularity(f, b, p=4, eps=1e-12, **quad_opts):
    """
    Integrate f(c) from c=0 to c=b with substitution c = u^p.
    Then dc = p*u^(p-1) du.
    """
    b = float(b)
    if b <= 0:
        return 0.0

    ub = b ** (1.0 / p)

    def g(u):
        c = u ** p
        return f(c) * (p * u ** (p - 1))

    # start at small u to avoid exactly 0
    return quad(g, eps, ub, **quad_opts)[0]


# ---------- Main computation ----------

def compute_expression_low_quota(theta: float, alpha: float, n: int,
                       eps: float = 1e-10,
                       quad_opts=None) -> float:
    """
    Computes:

      I2 = ∫_0^θ   P(c) * ( S(c)^2 / c^2 ) dc
         + ∫_θ^{1-θ} P(c) * ( (S(c) + 1/2^(n-1))^2 / c^2 ) dc
         + ∫_{1-θ}^1 P(c) * (1 / c^2) dc

      I1 = ∫_0^θ   P(c) * ( S(c) / c ) dc
         + ∫_θ^{1-θ} P(c) * ( (S(c) + 1/2^(n-1)) / c ) dc
         + ∫_{1-θ}^1 P(c) * (1 / c) dc

    returns: I2 - (I1)^2 (variance), I1 (mean)

    Notes:
    - Uses adaptive quadrature (scipy.integrate.quad).
    - Handles the c=0 endpoint by starting at eps.
    """
    if not (0.0 < theta < 0.5):
        raise ValueError("Require 0 < theta < 1/2 so that [theta, 1-theta] is non-empty.")
    if alpha <= 0:
        raise ValueError("alpha must be > 0.")
    if n < 2:
        raise ValueError("n must be >= 2 (and sums are meaningful for n >= 3).")

    if quad_opts is None:
        quad_opts = dict(limit=300, epsabs=1e-10, epsrel=1e-10)

    add_const = 1.0 / (2 ** (n - 1))

    # --- integrands for I2 ---
    def f2_region1(c):
        sc = S(c, theta, n, alpha)
        return px1(c, n, alpha) * (sc * sc) / (c * c)

    def f2_region2(c):
        sc = S(c, theta, n, alpha) + add_const
        return px1(c, n, alpha) * (sc * sc) / (c * c)

    def f2_region3(c):
        return px1(c, n, alpha) * (1.0 / (c * c))

    # --- integrands for I1 ---
    def f1_region1(c):
        sc = S(c, theta, n, alpha)
        return px1(c, n, alpha) * sc / c

    def f1_region2(c):
        sc = S(c, theta, n, alpha) + add_const
        return px1(c, n, alpha) * sc / c

    def f1_region3(c):
        return px1(c, n, alpha) * (1.0 / c)

    # Integrate with safe endpoints
    a1 = eps
    b1 = theta
    a2 = theta
    b2 = 1.0 - theta
    a3 = 1.0 - theta
    b3 = 1.0 - eps

    # for small c when alpha < 1
    small_c = 0.02

    # If theta is extremely small/large, guard empty intervals
    I2 = 0.0
    I1 = 0.0

    if alpha < 1:
        if theta <= 0.02:
            I2 += quad_zero_singularity(f2_region1, b1, p=4, eps=1e-12, **quad_opts)
            I1 += quad_zero_singularity(f1_region1, b1, p=4, eps=1e-12, **quad_opts)
        else:
            if small_c > a1:
                I2 += quad_zero_singularity(f2_region1, small_c, p=4, eps=1e-12, **quad_opts)
                I1 += quad_zero_singularity(f1_region1, small_c, p=4, eps=1e-12, **quad_opts)
                I2 += quad(f2_region1, small_c, b1, **quad_opts)[0]
                I1 += quad(f1_region1, small_c, b1, **quad_opts)[0]
            else:
                I2 += quad(f2_region1, a1, b1, **quad_opts)[0]
                I1 += quad(f1_region1, a1, b1, **quad_opts)[0]
    else:
        if b1 > a1:
            I2 += quad(f2_region1, a1, b1, **quad_opts)[0]
            I1 += quad(f1_region1, a1, b1, **quad_opts)[0]

    if b2 > a2:
        I2 += quad(f2_region2, a2, b2, **quad_opts)[0]
        I1 += quad(f1_region2, a2, b2, **quad_opts)[0]

    if b3 > a3:
        I2 += quad(f2_region3, a3, b3, **quad_opts)[0]
        I1 += quad(f1_region3, a3, b3, **quad_opts)[0]

    return I2 - (I1 ** 2), I1

def compute_expression_variant_high_quota(theta: float, alpha: float, n: int,
                               eps: float = 1e-10,
                               quad_opts=None) -> float:
    """
    Computes:

    I2 = ∫_0^{1-θ}   P(c) * ( S1(c)^2 / c^2 ) dc
       + ∫_{1-θ}^{θ} P(c) * ( S2(c)^2 / c^2 ) dc
       + ∫_{θ}^{1}   P(c) * ( 1 / c^2 ) dc

    I1 = ∫_0^{1-θ}   P(c) * ( S1(c) / c ) dc
       + ∫_{1-θ}^{θ} P(c) * ( S2(c) / c ) dc
       + ∫_{θ}^{1}   P(c) * ( 1 / c ) dc

    returns: I2 - (I1)^2 (variance), I1 (mean)

    IMPORTANT:
    The middle integral bounds [1-θ, θ] only make sense when θ >= 1/2.
    If θ < 1/2, this interval is empty (upper < lower) and the code will skip it.
    """
    if not (0.5 <= theta < 1.0):
        raise ValueError("Require 0.5 <= theta < 1.")
    if alpha <= 0:
        raise ValueError("alpha must be > 0.")
    if n < 2:
        raise ValueError("n must be >= 2 (and sums are meaningful for n >= 3).")

    if quad_opts is None:
        quad_opts = dict(limit=300, epsabs=1e-10, epsrel=1e-10)

    # integrands for I2
    def f2_r1(c):
        s = S(c, theta, n, alpha)
        return px1(c, n, alpha) * (s * s) / (c * c)

    def f2_r2(c):
        s = S2(c, theta, n, alpha)
        return px1(c, n, alpha) * (s * s) / (c * c)

    def f2_r3(c):
        return px1(c, n, alpha) * (1.0 / (c * c))

    # integrands for I1
    def f1_r1(c):
        s = S(c, theta, n, alpha)
        return px1(c, n, alpha) * s / c

    def f1_r2(c):
        s = S2(c, theta, n, alpha)
        return px1(c, n, alpha) * s / c

    def f1_r3(c):
        return px1(c, n, alpha) * (1.0 / c)

    # Bounds with endpoint safety
    # Region 1: [0, 1-θ]
    a1, b1 = eps, max(eps, 1.0 - theta)
    b1 = min(b1, 1.0 - eps)

    # Region 2: [1-θ, θ]  (may be empty depending on θ)
    a2, b2 = 1.0 - theta, theta

    # Region 3: [θ, 1]
    a3, b3 = max(theta, eps), 1.0 - eps

    I2 = 0.0
    I1 = 0.0

    # for alpha<1 and small c
    small_c = 0.02

    # Region 1
    if alpha < 1:
        if theta >= 0.02:
            I2 += quad_zero_singularity(f2_r1, b1, p=4, eps=1e-12, **quad_opts)
            I1 += quad_zero_singularity(f1_r1, b1, p=4, eps=1e-12, **quad_opts)
        else:
            if small_c > a1:
                I2 += quad_zero_singularity(f2_r1, small_c, p=4, eps=1e-12, **quad_opts)
                I1 += quad_zero_singularity(f1_r1, small_c, p=4, eps=1e-12, **quad_opts)
                I2 += quad(f2_r1, small_c, b1, **quad_opts)[0]
                I1 += quad(f1_r1, small_c, b1, **quad_opts)[0]
            else:
                I2 += quad(f2_r1, a1, b1, **quad_opts)[0]
                I1 += quad(f1_r1, a1, b1, **quad_opts)[0]
    else:
        if b1 > a1:
            I2 += quad(f2_r1, a1, b1, **quad_opts)[0]
            I1 += quad(f1_r1, a1, b1, **quad_opts)[0]

    # Region 2 (only if non-empty)
    if b2 > a2 + 1e-15:
        a2s = max(a2, eps)
        b2s = min(b2, 1.0 - eps)
        if b2s > a2s:
            I2 += quad(f2_r2, a2s, b2s, **quad_opts)[0]
            I1 += quad(f1_r2, a2s, b2s, **quad_opts)[0]

    # Region 3
    if b3 > a3:
        I2 += quad(f2_r3, a3, b3, **quad_opts)[0]
        I1 += quad(f1_r3, a3, b3, **quad_opts)[0]

    return I2 - (I1 ** 2), I1


##### parameters for lower quotas ######
alpha = 0.273568 # fit value is 0.273568
ns = [30, 40, 60, 80, 150]
# ns = [30]
theta_interval = 0.02
#######################

theta_number = int(0.5/theta_interval)

## lower quotas
low_quotas = [(i+1)*theta_interval for i in range(theta_number-1)]
var = {}
e = {}

for n in ns:
    var[str(n)] = []
    e[str(n)] = []
    for theta in low_quotas:
        val, ex = compute_expression_low_quota(theta=theta, alpha=alpha, n=n)
        var[str(n)].append(val)
        e[str(n)].append(ex)

    ## higher quotas
    high_quotas = [i*theta_interval+0.5 for i in range(theta_number)]
    for theta in high_quotas:
        val, ex = compute_expression_variant_high_quota(theta=theta, alpha=alpha, n=n)
        var[str(n)].append(val)
        e[str(n)].append(ex)
    # print("*********************")
    # print(var[str(n)])
    # print(e[str(n)])
    # breakpoint()

# print(low_var)
# breakpoint()

plt.figure()
for i in ns:
    plt.plot(low_quotas+high_quotas,var[str(i)], label = f"n={str(i)}")
plt.xlabel("quotas")
plt.ylabel("variance")
plt.legend()
plt.savefig(f"plots_analytical/variances_{str(alpha)}.pdf")
# breakpoint()
plt.figure()
for i in ns:
    plt.plot(low_quotas+high_quotas,e[str(i)], label = f"n={str(i)}")
plt.xlabel("quotas")
plt.ylabel("Mean")
plt.legend()
plt.savefig(f"plots_analytical/mean_{str(alpha)}.pdf")




# ---------- Example usage ----------
# if __name__ == "__main__":
#     theta = 0.2
#     alpha = 2.0
#     n = 100

#     val = compute_expression(theta=theta, alpha=alpha, n=n)
#     print("Value =", val)
