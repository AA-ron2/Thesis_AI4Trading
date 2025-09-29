import numpy as np

def hawkes_params(dt: float, p_target: float = 0.30, eta: float = 0.6, memory_steps: int = 10):
    """
    Pick Hawkes (mu, kappa, jump) from intuitive knobs:
      dt             : env step size
      p_target       : target mean per-step arrival probability per side
      eta in (0,1)   : branching ratio = jump/kappa (clustering strength)
      memory_steps   : ~how many steps until intensity decays substantially

    Returns:
      mu     : baseline arrival rate per side
      kappa  : mean-reversion speed
      jump   : excitation jump size
    """
    kappa = 1.0 / (memory_steps * dt)           # decay time ≈ memory_steps * dt
    jump  = eta * kappa                         # ensure branching ratio eta = jump/kappa
    lam_bar = p_target / dt                     # desired mean intensity
    mu = (1.0 - eta) * lam_bar                  # stationarity: E[lambda] = mu / (1 - eta)
    return mu, kappa, jump

def glft_constants(gamma: float, A: float, k: float, xi: float, tick: float):
    """
    GLFT constants c1, c2 (see eqs. (4.6)-(4.7) in Guéant-Lehalle-Fernandez-Tapia).
    c1 has units of 'price', c2 ~ 1/sigma (we multiply by sigma later).
    """
    xi, tick = float(xi), float(tick)
    k = float(k); A = float(A); gamma = float(gamma)

    # c1 = (1/(xi*Δ)) * ln(1 + xi*Δ/k)
    c1 = (1.0 / (xi * tick)) * np.log(1.0 + (xi * tick) / k)

    # c2 = sqrt( gamma / (2 A Δ k) * (1 + xi*Δ/k)^(k/(xi*Δ) + 1) )
    c2 = np.sqrt(
        (gamma / (2.0 * A * tick * k)) *
        (1.0 + (xi * tick) / k) ** (k / (xi * tick) + 1.0)
    )
    return c1, c2


def glft_half_spreads(q: np.ndarray, sigma: float, gamma: float, A: float, k: float,
                      xi: float = 1.0, tick: float = 1.0) -> np.ndarray:
    """
    Returns [bid_half, ask_half] using GLFT approximate quotes:
      δ_b = c1 + (Δ/2) σ c2 + q σ c2
      δ_a = c1 + (Δ/2) σ c2 - q σ c2
    (Half-spread = c1 + (Δ/2)σ c2, Skew = σ c2)
    """
    c1, c2 = glft_constants(gamma=gamma, A=A, k=k, xi=xi, tick=tick)
    half = c1 + 0.5 * tick * float(sigma) * c2
    skew = float(sigma) * c2

    q = np.asarray(q, dtype=float).reshape(-1)
    bid_half = np.maximum(0.0, half + skew * q)
    ask_half = np.maximum(0.0, half - skew * q)
    return np.stack([bid_half, ask_half], axis=1)


def fit_exponential_intensity(delta: np.ndarray, p_step: np.ndarray, dt: float, eps: float = 1e-12):
    """
    Fit λ(δ) = A * exp(-k δ) from per-step execution probabilities p(δ).
    Uses log(λ) = log(p/dt) ≈ log A - k δ (linear regression).
    WARNING: ignores queue position / sign issues; use carefully.
    """
    delta = np.asarray(delta, float).reshape(-1)
    lam_hat = np.maximum(p_step, eps) / float(dt)
    y = np.log(lam_hat)                # ≈ log A - k * δ
    X = np.vstack([np.ones_like(delta), -delta]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    logA, k = beta[0], beta[1]
    A = float(np.exp(logA))
    return A, float(k)
