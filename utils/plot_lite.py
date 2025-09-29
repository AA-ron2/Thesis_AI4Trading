# utils/plotting_lite.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.calibration import glft_constants

# Optional; comment out if you don't want seaborn
import seaborn as sns

# Indices for our env's observation layout
ASSET_PRICE, INVENTORY, TIMEIDX, CASH = 0, 1, 2, 3


# ---------- BATCHED ROLLOUT (mbt-gym style tensors) ----------
def generate_trajectory_lite(env: gym.Env, agent, seed: int = None, include_log_probs: bool = False):
    """
    Returns:
      observations: (N, obs_dim, T+1)
      actions:      (N, act_dim, T)
      rewards:      (N, 1, T)
      (optional) log_probs: (N, act_dim, T) if agent returns (action, log_prob)
    """
    if seed is not None:
        env.reset(seed=seed)

    # Safety: if you want full batch logging, env should be vectorized
    if not getattr(env, "return_vectorized", False):
        assert getattr(env, "N", 1) == 1, "For batch logging across N>1, set env.return_vectorized=True."

    obs, _ = env.reset()
    # Normalize shapes
    if obs.ndim == 1:
        # single trajectory view
        N = 1
        obs_dim = obs.shape[0]
    else:
        N = obs.shape[0]
        obs_dim = obs.shape[1]

    act_dim = env.action_space.shape[0]
    T = env.M

    observations = np.zeros((N, obs_dim, T + 1), dtype=np.float32)
    actions      = np.zeros((N, act_dim, T), dtype=np.float32)
    rewards      = np.zeros((N, 1, T), dtype=np.float32)
    log_probs    = np.zeros((N, act_dim, T), dtype=np.float32) if include_log_probs else None

    # t = 0 observation
    if N == 1 and obs.ndim == 1:
        observations[0, :, 0] = obs
    else:
        observations[:, :, 0] = obs

    t = 0
    while True:
        # Query agent
        if include_log_probs:
            out = agent.get_action(obs)
            if isinstance(out, tuple) and len(out) == 2:
                action, logp = out
            else:
                action, logp = out, None
        else:
            action = agent.get_action(obs)

        # Step env
        step_out = env.step(action)
        if len(step_out) == 5:
            obs_next, r, done, truncated, info = step_out
        else:
            # Gym v0 fallback
            obs_next, r, done, info = step_out
            truncated = False

        # Store
        if N == 1 and np.asarray(action).ndim == 1:
            actions[0, :, t] = np.asarray(action, dtype=np.float32)
            rewards[0, 0, t] = float(r)
            observations[0, :, t + 1] = obs_next
            if include_log_probs and log_probs is not None and logp is not None:
                log_probs[0, :, t] = np.asarray(logp, dtype=np.float32)
        else:
            actions[:, :, t] = np.asarray(action, dtype=np.float32)
            rewards[:, 0, t] = np.asarray(r, dtype=np.float32).reshape(-1)
            observations[:, :, t + 1] = obs_next
            if include_log_probs and log_probs is not None and logp is not None:
                log_probs[:, :, t] = np.asarray(logp, dtype=np.float32)

        # Termination
        if isinstance(done, np.ndarray):
            if done.all():
                break
        else:
            if done:
                break

        t += 1
        if t >= T:
            break
        obs = obs_next

    return (observations, actions, rewards, log_probs) if include_log_probs else (observations, actions, rewards)


# ---------- PLOTTING ----------
def get_timestamps(env):
    # time in "continuous" units like mbt-gym
    return np.linspace(0.0, env.T, env.M + 1)
    
# def plot_trajectory(env: gym.Env, agent, seed: int = None):
#     """
#     Overlays on the midprice panel:
#       - mid price
#       - quoted bid/ask (from actions)
#       - reservation price r_t
#       - reservation bid/ask r_t ± spread/2
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     ASSET_PRICE, INVENTORY, TIMEIDX, CASH = 0, 1, 2, 3

#     # --- rollout (mbt-gym style shapes) ---
#     timestamps = np.linspace(0.0, env.T, env.M + 1)
#     observations, actions, rewards = generate_trajectory_lite(env, agent, seed)
#     N, obs_dim, T1 = observations.shape
#     _, act_dim, T = actions.shape  # T1 = T+1

#     # unpack
#     S = observations[:, ASSET_PRICE, :]          # (N, T+1)
#     q = observations[:, INVENTORY, :]            # (N, T+1)
#     t_idx = observations[:, TIMEIDX, :]          # (N, T+1) integer index
#     t_cont = t_idx * env.dt                      # (N, T+1)
#     hb = actions[:, 0, :]                        # (N, T)
#     ha = actions[:, 1, :] if act_dim > 1 else np.zeros_like(hb)

#     # quoted bid/ask use the *post-step* price S[:, 1:]
#     bid_quoted = S[:, 1:] - hb                   # (N, T)
#     ask_quoted = S[:, 1:] + ha                   # (N, T)

#     # --- reservation price (two ways) ---
#     # Try "theoretical" AS if agent exposes gamma (or risk_aversion); else derive from actions.
#     gamma = getattr(agent, "gamma", getattr(agent, "risk_aversion", None))
#     sigma = getattr(env.dyn.mid, "sigma", None)
#     k_fill = getattr(env.dyn, "fill_k", None)

#     if (gamma is not None) and (sigma is not None) and (k_fill is not None):
#         # AS reservation price r_t = S_t - q_t * gamma * sigma^2 * (T - t)
#         adj = gamma * (sigma ** 2) * (env.T - t_cont)            # (N, T+1)
#         r_theo = S - q * adj                                     # (N, T+1)

#         # AS half-spread: 0.5 * [gamma*sigma^2*(T - t) + (2/gamma) * log(1 + gamma/k)]
#         if gamma == 0:
#             half_spread_theo = np.full_like(r_theo, 1.0 / k_fill)
#         else:
#             vol_term  = gamma * (sigma ** 2) * (env.T - t_cont)
#             fill_term = (2.0 / gamma) * np.log(1.0 + gamma / k_fill)
#             half_spread_theo = 0.5 * (vol_term + fill_term)

#         res_bid = r_theo - half_spread_theo                      # (N, T+1)
#         res_ask = r_theo + half_spread_theo                      # (N, T+1)
#         show_reservation_from = "AS (theoretical)"
#     else:
#         # Fallback: define r_t from quoted halves on post-step grid (length T), then pad to T+1
#         # r = S_{t+1} + (ha - hb)/2; spread_half = (hb + ha)/2
#         r_step = S[:, 1:] + 0.5 * (ha - hb)                      # (N, T)
#         half_spread_step = 0.5 * (hb + ha)                       # (N, T)
#         res_bid_step = r_step - half_spread_step                 # == bid_quoted
#         res_ask_step = r_step + half_spread_step                 # == ask_quoted

#         # pad a leading NaN to align with T+1 for plotting over the same timestamps
#         pad = np.full((N, 1), np.nan, dtype=float)
#         r_theo  = np.concatenate([pad, r_step], axis=1)          # (N, T+1)
#         res_bid = np.concatenate([pad, res_bid_step], axis=1)
#         res_ask = np.concatenate([pad, res_ask_step], axis=1)
#         show_reservation_from = "derived from quotes"

#     # --- rewards cum ---
#     rewards_squeezed = np.squeeze(rewards, axis=1)               # (N, T)
#     cum_rewards = np.cumsum(rewards_squeezed, axis=-1)

#     # --- plot ---
#     colors = ["r", "k", "b", "g", "m", "c"]
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
#     ax3a = ax3.twinx()

#     ax1.set_title("Cumulative rewards")
#     ax2.set_title(f"Mid, Quotes, Reservation (source: {show_reservation_from})")
#     ax3.set_title("Inventory (L) & Cash (R)")
#     ax4.set_title("Actions")

#     cash = observations[:, CASH, :]

#     for i in range(N):
#         alpha = (i + 1) / (N + 1)

#         # cum rewards
#         ax1.plot(timestamps[1:], cum_rewards[i, :], alpha=alpha)

#         # mid (T+1)
#         ax2.plot(timestamps, S[i, :], color="k", lw=1.5, alpha=alpha, label="Mid" if i == 0 else None)
#         # quoted bid/ask (T)
#         ax2.plot(timestamps[1:], bid_quoted[i, :], color="tab:blue", alpha=alpha, label="Quoted Bid" if i == 0 else None)
#         ax2.plot(timestamps[1:], ask_quoted[i, :], color="tab:orange", alpha=alpha, label="Quoted Ask" if i == 0 else None)
#         # reservation price (T+1)
#         ax2.plot(timestamps, r_theo[i, :], color="tab:green", ls="--", alpha=alpha, label="Reservation price" if i == 0 else None)
#         # reservation bid/ask (T+1)
#         ax2.plot(timestamps, res_bid[i, :], color="tab:blue", ls="--", alpha=alpha, label="Reservation Bid" if i == 0 else None)
#         ax2.plot(timestamps, res_ask[i, :], color="tab:orange", ls="--", alpha=alpha, label="Reservation Ask" if i == 0 else None)

#         # inventory & cash (T+1)
#         ax3.plot(timestamps, q[i, :], color="r", alpha=alpha, label="Inventory" if i == 0 else None)
#         ax3a.plot(timestamps, cash[i, :], color="b", alpha=alpha, label="Cash" if i == 0 else None)

#         # actions (T)
#         for j in range(act_dim):
#             ax4.plot(timestamps[:-1], actions[i, j, :],
#                      color=colors[j % len(colors)], alpha=alpha,
#                      label=(f"action[{j}]" if (i == 0) else None))

#     # legends & cosmetics
#     for ax in (ax1, ax2, ax3, ax4):
#         ax.grid(True, alpha=0.3)
#     ax2.legend(loc="best")
#     h3, l3 = ax3.get_legend_handles_labels()
#     h3a, l3a = ax3a.get_legend_handles_labels()
#     if h3 or h3a:
#         ax3.legend(h3 + h3a, l3 + l3a, loc="best")
#     ax4.legend(loc="best")

#     plt.tight_layout()
#     plt.show()

def plot_trajectory(env: gym.Env, agent, seed: int = None, show_reservation: bool = True):
    """
    Panels:
      (1) cumulative rewards
      (2) mid + quoted bid/ask + (optional) theoretical reservation bid/ask & reservation mid
      (3) inventory & cash
      (4) actions
    Auto-select reservation model by agent.mode ('finite' or 'infinite').
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # bring in rollout helper from your file/module if needed
    # from utils.plotting_lite import generate_trajectory_lite

    # try to import your infinite helper (adjust path if different)
    try:
        from agents.avellaneda_stoikov import as_infinite_half_spreads as _as_inf_halves
    except Exception:
        _as_inf_halves = None

    ASSET_PRICE, INVENTORY, TIMEIDX, CASH = 0, 1, 2, 3

    # --- rollout ---
    timestamps = np.linspace(0.0, env.T, env.M + 1)
    observations, actions, rewards = generate_trajectory_lite(env, agent, seed)
    N, _, T1 = observations.shape
    _, act_dim, T = actions.shape  # T1 == T+1
    assert act_dim >= 2, "Actions must contain at least [bid_half, ask_half]."

    # unpack
    S = observations[:, ASSET_PRICE, :]            # (N, T+1)
    q = observations[:, INVENTORY, :]              # (N, T+1)
    t_idx = observations[:, TIMEIDX, :]            # (N, T+1)
    cash = observations[:, CASH, :]
    hb = actions[:, 0, :]                          # (N, T)
    ha = actions[:, 1, :] if act_dim > 1 else np.zeros_like(hb)

    # quoted bid/ask live on the post-step grid (env updates price first in step)
    bid_quoted = S[:, 1:] - hb                     # (N, T)
    ask_quoted = S[:, 1:] + ha                     # (N, T)

    # rewards → cumulative
    r_step = np.squeeze(rewards, axis=1)           # (N, T)
    cum_rewards = np.cumsum(r_step, axis=-1)

    # ---- reservation overlays (optional) ----
    res_bid = res_ask = r_mid = None
    res_source = None
    if show_reservation:
        mode = str(getattr(agent, "mode", "finite")).lower()
        gamma = getattr(agent, "gamma", getattr(agent, "risk_aversion", None))
        sigma = getattr(env.dyn.mid, "sigma", None)

        if mode == "finite" and (gamma is not None) and (sigma is not None):
            k = float(getattr(agent, "k", getattr(env.dyn, "fill_k", None)))
            if k is not None:
                sigma2 = float(sigma) ** 2
                Ttot = float(env.T)
                t = t_idx * env.dt                           # (N, T+1)
                r_mid = S - q * gamma * sigma2 * (Ttot - t)  # (N, T+1)
                if gamma == 0.0:
                    half = np.full_like(r_mid, 1.0 / k)
                else:
                    half = 0.5 * (gamma * sigma2 * (Ttot - t) + (2.0 / gamma) * np.log(1.0 + gamma / k))
                res_bid = r_mid - half
                res_ask = r_mid + half
                res_source = "AS finite (theoretical)"

        elif mode == "infinite" and (gamma is not None) and (sigma is not None) and (_as_inf_halves is not None):
            omega = getattr(agent, "omega", None)
            if omega is not None:
                # ALIGNMENT: use q at time t (pre-step) and S at time t+1 (post-step), then pad to match T+1
                S_post = S[:, 1:]           # (N, T)
                q_pre  = q[:, :-1]          # (N, T)
                s_flat = S_post.reshape(-1)
                q_flat = q_pre.reshape(-1)
                halves_flat = _as_inf_halves(s_flat, q_flat, float(gamma), float(sigma), float(omega))  # (N*T, 2)
                halves = halves_flat.reshape(N, T, 2)
                bid_half_step = halves[:, :, 0]
                ask_half_step = halves[:, :, 1]
                res_bid_step = S_post - bid_half_step           # (N, T)
                res_ask_step = S_post + ask_half_step           # (N, T)
                r_mid_step  = 0.5 * (res_bid_step + res_ask_step)

                pad = np.full((N, 1), np.nan, dtype=float)
                res_bid = np.concatenate([pad, res_bid_step], axis=1)   # (N, T+1)
                res_ask = np.concatenate([pad, res_ask_step], axis=1)   # (N, T+1)
                r_mid   = np.concatenate([pad, r_mid_step], axis=1)     # (N, T+1)
                res_source = "AS infinite (theoretical)"
        
        elif mode == "glft":
            # reservation mid for GLFT: r = S - (skew * q)
            # skew = σ c2 ; half = c1 + (Δ/2) σ c2
            c1, c2 = glft_constants(agent.gamma, agent.A, agent.k, agent.xi, agent.tick)
            skew = agent.sigma * c2
            half = c1 + 0.5 * agent.tick * agent.sigma * c2

            # (T+1)-length arrays: use pre-step q (q[:, :-1]) with post-step S (S[:, 1:]),
            # then pad a NaN on the left to align to T+1 timeline like the rest of the plot.
            S_post = S[:, 1:]
            q_pre  = q[:, :-1]
            res_bid_step = S_post - (half + skew * q_pre)
            res_ask_step = S_post + (half - skew * q_pre)
            r_mid_step   = 0.5 * (res_bid_step + res_ask_step)

            pad = np.full((S.shape[0], 1), np.nan)
            res_bid = np.concatenate([pad, res_bid_step], axis=1)
            res_ask = np.concatenate([pad, res_ask_step], axis=1)
            r_mid   = np.concatenate([pad, r_mid_step],   axis=1)
            res_source = "GLFT (theoretical)"

    # ---- plotting ----
    colors = ["r", "k", "b", "g", "m", "c"]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    ax3a = ax3.twinx()

    ax1.set_title("Cumulative rewards")
    ax2.set_title("Mid, Quotes" + (f", Reservation ({res_source})" if res_source else ""))
    ax3.set_title("Inventory (L) & Cash (R)")
    ax4.set_title("Actions")

    for i in range(N):
        alpha = (i + 1) / (N + 1)

        # cum rewards
        ax1.plot(timestamps[1:], cum_rewards[i, :], alpha=alpha)

        # mid
        ax2.plot(timestamps, S[i, :], color="k", lw=1.5, alpha=alpha, label="Mid" if i == 0 else None)

        # quoted bid/ask from actions (always shown)
        ax2.plot(timestamps[1:], bid_quoted[i, :], color="tab:blue", alpha=alpha,
                 label="Quoted Bid" if i == 0 else None)
        ax2.plot(timestamps[1:], ask_quoted[i, :], color="tab:orange", alpha=alpha,
                 label="Quoted Ask" if i == 0 else None)

        # reservation overlays (if computed)
        if res_bid is not None:
            ax2.plot(timestamps, res_bid[i, :], color="tab:blue", ls="--", alpha=alpha,
                     label="Reservation Bid" if i == 0 else None)
        if res_ask is not None:
            ax2.plot(timestamps, res_ask[i, :], color="tab:orange", ls="--", alpha=alpha,
                     label="Reservation Ask" if i == 0 else None)
        if r_mid is not None:
            ax2.plot(timestamps, r_mid[i, :], color="tab:green", ls="--", alpha=alpha,
                     label="Reservation mid" if i == 0 else None)

        # inventory & cash
        ax3.plot(timestamps, q[i, :], color="r", alpha=alpha, label="Inventory" if i == 0 else None)
        ax3a.plot(timestamps, cash[i, :], color="b", alpha=alpha, label="Cash" if i == 0 else None)

        # actions
        for j in range(act_dim):
            ax4.plot(timestamps[:-1], actions[i, j, :],
                     color=colors[j % len(colors)], alpha=alpha,
                     label=(f"action[{j}]" if (i == 0) else None))

    # legends
    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    h3, l3 = ax3.get_legend_handles_labels()
    h3a, l3a = ax3a.get_legend_handles_labels()
    if h3 or h3a:
        ax3.legend(h3 + h3a, l3 + l3a, loc="best")
    ax4.legend(loc="best")

    plt.tight_layout()
    plt.show()


# ---------- STABLE-BASELINES ACTION SCAN (optional) ----------
def plot_stable_baselines_actions(model, env, inventories=range(-3, 4), price=100.0, cash=0.0):
    """
    Scans model actions over time for fixed (price, cash) and various inventories.
    NOTE: our obs order is [price, inventory, time_idx, cash].
    """
    ts = np.arange(env.M + 1)
    inv_action_map = {}

    for q in inventories:
        # t = 0
        obs = np.array([price, q, 0, cash], dtype=np.float32)
        act = model.predict(obs, deterministic=True)[0].reshape((1, -1))
        for t in ts[1:]:
            obs = np.array([price, q, t, cash], dtype=np.float32)
            a = model.predict(obs, deterministic=True)[0].reshape((1, -1))
            act = np.vstack([act, a])
        inv_action_map[q] = act  # (M+1, act_dim)

    # plot first two dims (half-spreads)
    for dim, title in zip([0, 1], ["Bid half-spread", "Ask half-spread"]):
        plt.figure()
        for q in inventories:
            plt.plot(inv_action_map[q][:, dim], label=f"q={q}")
        plt.title(f"SB3 policy: {title} vs time_idx")
        plt.xlabel("time_idx"); plt.ylabel("half-spread")
        plt.legend(); plt.grid(True, alpha=0.3)
    plt.show()


# ---------- PnL HIST + SUMMARY ----------
def plot_pnl(rewards, symmetric_rewards=None):
    """
    rewards: array-like of episode PnL (sum over time), shape (N,)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    if symmetric_rewards is not None:
        sns.histplot(symmetric_rewards, label="Symmetric strategy", stat="density", bins=50, ax=ax)
    sns.histplot(rewards, label="Rewards", color="red", stat="density", bins=50, ax=ax)
    ax.legend(); ax.set_title("Episode PnL distribution"); ax.set_xlabel("PnL"); ax.set_ylabel("Density")
    plt.tight_layout()
    return fig


def generate_results_table_and_hist(env: gym.Env, agent, n_episodes: int = 1):
    """
    Requires env.return_vectorized=True and N>1 to produce cross-sectional stats from one episode.
    (Set a large N for tight confidence bands, e.g., N=512+.)
    """
    assert getattr(env, "return_vectorized", False) and env.N > 1, \
        "Set env.return_vectorized=True with N>1 to generate batch results."

    observations, actions, rewards = generate_trajectory_lite(env, agent)
    # shapes:
    # observations: (N, 4, T+1), actions: (N, A, T), rewards: (N, 1, T)
    N, _, Tp1 = observations.shape
    T = Tp1 - 1
    total_rewards = rewards.sum(axis=-1).reshape(N)                  # (N,)
    terminal_inventories = observations[:, INVENTORY, -1]            # (N,)

    # Use only the first two action dims (half-spreads) to define total spread
    half_spreads_mean = actions[:, :2, :].mean(axis=(1, 2))          # (N,)
    mean_spread = 2.0 * np.mean(half_spreads_mean)

    rows = ["Inventory"]
    columns = ["Mean spread", "Mean PnL", "Std PnL", "Mean terminal inventory", "Std terminal inventory"]
    results = pd.DataFrame(index=rows, columns=columns)
    results.loc[:, "Mean spread"] = mean_spread
    results.loc["Inventory", "Mean PnL"] = float(np.mean(total_rewards))
    results.loc["Inventory", "Std PnL"] = float(np.std(total_rewards))
    results.loc["Inventory", "Mean terminal inventory"] = float(np.mean(terminal_inventories))
    results.loc["Inventory", "Std terminal inventory"] = float(np.std(terminal_inventories))

    fig = plot_pnl(total_rewards)
    return results, fig, total_rewards