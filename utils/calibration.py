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
