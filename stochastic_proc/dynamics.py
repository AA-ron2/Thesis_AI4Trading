from __future__ import annotations
import numpy as np
from typing import Optional
import gymnasium as gym

FILL_MULT = np.array([[-1.0, +1.0]], dtype=float)  # bid=-1, ask=+1; broadcast to (N,2)

class LimitOrderDynamics:
    """
    Binds processes + defines action semantics for LIMIT orders:
    action = [bid_half_spread, ask_half_spread], both >= 0
    """
    def __init__(self, mid_model, arr_model, fill_k: float, max_depth: Optional[float] = None):
        self.mid = mid_model
        self.arr = arr_model
        self.fill_k = float(fill_k)
        self.max_depth = float(max_depth) if max_depth is not None else np.inf
        self.num_traj = self.mid.num_traj
        self.dt = self.mid.dt

    def get_action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=0.0, high=self.max_depth, shape=(2,), dtype=np.float32)

    @property
    def midprice(self) -> np.ndarray:
        return self.mid.state[:, [0]]  # (N,1) for clean broadcasting

    def arrivals_and_fills(self, half_spreads: np.ndarray, rng: np.random.Generator):
        # arrivals
        out = self.arr.step()
        arrivals = out["arrivals"].astype(bool)                    # (N,2)
        # fill probs given arrival
        hs = np.maximum(half_spreads, 0.0)
        p_fill = np.exp(-self.fill_k * hs)
        u = rng.uniform(size=p_fill.shape)
        fills = arrivals & (u < p_fill)                            # bool (N,2)
        return arrivals, fills

    def cash_inventory_delta(self, half_spreads: np.ndarray, arrivals: np.ndarray, fills: np.ndarray):
        # prices on each side: mid +/- half_spread  (using FILL_MULT)
        per_side_px = self.midprice + half_spreads * FILL_MULT     # (N,2)
        dq = np.sum(arrivals * fills * (-FILL_MULT), axis=1)       # (N,)
        dcash = np.sum(FILL_MULT * arrivals * fills * per_side_px, axis=1)  # (N,)
        return dq.astype(int), dcash.astype(float)

class LimitAndMarketDynamics(LimitOrderDynamics):
    """
    Adds immediate market orders:
    action = [bid_half, ask_half, mo_buy_flag, mo_sell_flag] in [0,1] for flags (>0.5 executes)
    Executes market orders at fixed half-spread around mid.
    """
    def __init__(self, mid_model, arr_model, fill_k: float, fixed_mo_half_spread: float = 0.5, max_depth: Optional[float] = None):
        super().__init__(mid_model, arr_model, fill_k, max_depth)
        self.fixed_mo_half = float(fixed_mo_half_spread)

    def get_action_space(self) -> gym.spaces.Box:
        high = np.array([self.max_depth, self.max_depth, 1.0, 1.0], dtype=np.float32)
        low  = np.zeros_like(high, dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, shape=(4,), dtype=np.float32)

    def market_order_deltas(self, action: np.ndarray):
        mo_buy  = (action[:, 2] > 0.5).astype(int)
        mo_sell = (action[:, 3] > 0.5).astype(int)
        best_bid = (self.midprice[:, 0] - self.fixed_mo_half)      # we SELL at best_bid
        best_ask = (self.midprice[:, 0] + self.fixed_mo_half)      # we BUY at best_ask
        dq = mo_buy - mo_sell
        dcash = (mo_sell * best_bid) - (mo_buy * best_ask)
        return dq, dcash

class HistoricalLimitOrderDynamics:
    """
    Deterministic fills using realized L2 snapshots.
    action = [bid_half_spread, ask_half_spread] (>=0).
    Fills when observed trades cross our quote (conservative).
    """
    def __init__(self, feed, tick_size: float, quote_size: float = 1.0, max_depth: float = 20.0, num_traj: int = 1):
        self.feed = feed
        self.tick = float(tick_size)
        self.qty  = float(quote_size)
        self.max_depth = float(max_depth)
        self.num_traj = int(num_traj)
        # “mid” comes from the feed each step; keep dt through your env.

    def get_action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=0.0, high=self.max_depth, shape=(2,), dtype=np.float32)

    @property
    def midprice(self) -> np.ndarray:
        # env reads mid from its mid process; for convenience compute here too
        s = self.feed.snapshot()
        mid = np.atleast_1d(s["mid"]).astype(float)
        return mid.reshape(-1, 1)

    def arrivals_and_fills(self, half_spreads: np.ndarray, rng):
        # no stochastic arrivals; we compute fills directly
        return None, None

    def cash_inventory_delta(self, half_spreads: np.ndarray, arrivals, fills):
        s = self.feed.snapshot()
        mid = np.atleast_1d(s["mid"]).astype(float)              # (N,)
        bb  = np.atleast_1d(s["best_bid"]).astype(float)
        ba  = np.atleast_1d(s["best_ask"]).astype(float)

        hb = half_spreads[:, 0]
        ha = half_spreads[:, 1]
        # clip quotes inside the spread (don’t cross)
        bid_px = np.minimum(ba - self.tick, mid - hb)
        ask_px = np.maximum(bb + self.tick, mid + ha)

        fill_bid = np.zeros(self.num_traj, float)
        fill_ask = np.zeros(self.num_traj, float)

        if ("trade_px" in s) and (s["trade_px"] is not None):
            px  = np.atleast_1d(s["trade_px"]).astype(float)
            sz  = np.atleast_1d(s["trade_sz"]).astype(float)
            side= np.atleast_1d(s["trade_side"]).astype(float)   # +1 buy MO, -1 sell MO
            # conservative: fill if MO price crosses our quote
            fill_bid = np.where((side < 0) & (px <= bid_px), np.minimum(self.qty, sz), 0.0)
            fill_ask = np.where((side > 0) & (px >= ask_px), np.minimum(self.qty, sz), 0.0)
        # else: keep zero fills (or add BBO-move heuristic if desired)

        dq    = (fill_bid - fill_ask).astype(int)
        dcash = (-fill_bid * bid_px + fill_ask * ask_px).astype(float)
        return dq, dcash
