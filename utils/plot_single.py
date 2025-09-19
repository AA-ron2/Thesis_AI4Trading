# utils/plot_single.py
import numpy as np
import matplotlib.pyplot as plt

ASSET_PRICE, INVENTORY, TIMEIDX, CASH = 0, 1, 2, 3

def run_and_log(env, agent, episode_seed=None):
    obs, _ = env.reset(seed=episode_seed)
    T = env.M

    logs = {
        "t": np.arange(T, dtype=int),
        "price": np.empty(T, float),
        "bid": np.empty(T, float),
        "ask": np.empty(T, float),
        "half_bid": np.empty(T, float),
        "half_ask": np.empty(T, float),
        "q": np.empty(T, int),
        "cash": np.empty(T, float),
        "value": np.empty(T, float),
        "pnl_step": np.empty(T, float),
    }

    done = False
    t = 0
    while not done:
        # action for current state (half-spreads)
        action = agent.get_action(obs)               # shape (2,)
        # step
        obs_next, r, done, _, info = env.step(action)

        # log AFTER step (aligned to t index)
        price = obs_next[ASSET_PRICE]
        q     = int(obs_next[INVENTORY])
        cash  = float(obs_next[CASH])

        logs["price"][t]    = price
        logs["q"][t]        = q
        logs["cash"][t]     = cash
        logs["value"][t]    = q * price + cash
        logs["pnl_step"][t] = r

        hb, ha = float(action[0]), float(action[1])
        logs["half_bid"][t] = hb
        logs["half_ask"][t] = ha
        logs["bid"][t]      = price - hb
        logs["ask"][t]      = price + ha

        obs = obs_next
        t += 1
        if t >= T: break

    return logs

def plot_single_episode(logs, title_prefix=""):
    t = logs["t"]
    # 1) Price & quotes
    plt.figure()
    plt.plot(t, logs["price"], label="Mid")
    plt.plot(t, logs["bid"],   label="Bid")
    plt.plot(t, logs["ask"],   label="Ask")
    plt.legend()
    plt.title(f"{title_prefix} Price & Quotes")
    plt.xlabel("Step"); plt.ylabel("Price")

    # 2) Inventory
    plt.figure()
    plt.plot(t, logs["q"])
    plt.title(f"{title_prefix} Inventory")
    plt.xlabel("Step"); plt.ylabel("Units")

    # 3) Portfolio value (mark-to-market)
    plt.figure()
    plt.plot(t, logs["value"])
    plt.title(f"{title_prefix} Portfolio Value")
    plt.xlabel("Step"); plt.ylabel("Value")

    # 4) Step PnL
    plt.figure()
    plt.plot(t, logs["pnl_step"])
    plt.title(f"{title_prefix} Step PnL")
    plt.xlabel("Step"); plt.ylabel("PnL")

    plt.show()
