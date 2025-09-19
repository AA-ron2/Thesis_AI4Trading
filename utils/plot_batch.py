# utils/plot_batch.py
import numpy as np
import matplotlib.pyplot as plt

ASSET_PRICE, INVENTORY, TIMEIDX, CASH = 0, 1, 2, 3

def simulate_batch(env, agent):
    """Run one episode with env.return_vectorized=True; returns dict of arrays:
       price(T,N), q(T,N), value(T,N), hb(T,N), ha(T,N), bid(T,N), ask(T,N)"""
    assert env.return_vectorized, "Enable return_vectorized=True for batch simulation."
    obs, _ = env.reset()
    T, N = env.M, env.N

    price = np.empty((T, N), float)
    q     = np.empty((T, N), int)
    cash  = np.empty((T, N), float)
    value = np.empty((T, N), float)
    hb    = np.empty((T, N), float)
    ha    = np.empty((T, N), float)
    bid   = np.empty((T, N), float)
    ask   = np.empty((T, N), float)

    done = np.array([False]*N)
    t = 0
    while not done.all():
        action = agent.get_action(obs)  # shape (N,2) for AS
        obs_next, r, done, _, info = env.step(action)

        price[t, :] = obs_next[:, ASSET_PRICE]
        q[t, :]     = obs_next[:, INVENTORY].astype(int)
        cash[t, :]  = obs_next[:, CASH]
        value[t, :] = q[t, :] * price[t, :] + cash[t, :]

        hb[t, :] = action[:, 0]
        ha[t, :] = action[:, 1]
        bid[t, :] = price[t, :] - hb[t, :]
        ask[t, :] = price[t, :] + ha[t, :]

        obs = obs_next
        t += 1
        if t >= T: break

    return {"price": price, "q": q, "cash": cash, "value": value,
            "half_bid": hb, "half_ask": ha, "bid": bid, "ask": ask}

def _band(yTn, q_low=5.0, q_high=95.0):
    """mean and percentile bands over N"""
    mean = np.nanmean(yTn, axis=1)
    low  = np.nanpercentile(yTn, q_low, axis=1)
    high = np.nanpercentile(yTn, q_high, axis=1)
    return mean, low, high

def plot_batch(summary, title_prefix="Batch"):
    T = summary["price"].shape[0]
    t = np.arange(T)

    # 1) Mid + quotes
    m_mid, l_mid, h_mid = _band(summary["price"])
    m_bid, l_bid, h_bid = _band(summary["bid"])
    m_ask, l_ask, h_ask = _band(summary["ask"])
    plt.figure()
    plt.plot(t, m_mid, label="Mid (mean)")
    plt.fill_between(t, l_mid, h_mid, alpha=0.2, label="Mid (5–95%)")
    plt.plot(t, m_bid, label="Bid (mean)")
    plt.fill_between(t, l_bid, h_bid, alpha=0.2, label="Bid (5–95%)")
    plt.plot(t, m_ask, label="Ask (mean)")
    plt.fill_between(t, l_ask, h_ask, alpha=0.2, label="Ask (5–95%)")
    plt.legend()
    plt.title(f"{title_prefix}: Quotes")
    plt.xlabel("Step"); plt.ylabel("Price")

    # 2) Inventory
    m_q, l_q, h_q = _band(summary["q"])
    plt.figure()
    plt.plot(t, m_q, label="q (mean)")
    plt.fill_between(t, l_q, h_q, alpha=0.2, label="q (5–95%)")
    plt.legend()
    plt.title(f"{title_prefix}: Inventory")
    plt.xlabel("Step"); plt.ylabel("Units")

    # 3) Portfolio value
    m_v, l_v, h_v = _band(summary["value"])
    plt.figure()
    plt.plot(t, m_v, label="Value (mean)")
    plt.fill_between(t, l_v, h_v, alpha=0.2, label="Value (5–95%)")
    plt.legend()
    plt.title(f"{title_prefix}: Portfolio Value")
    plt.xlabel("Step"); plt.ylabel("Value")

    plt.show()