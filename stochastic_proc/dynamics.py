from __future__ import annotations
import numpy as np
from typing import Optional
import gymnasium as gym
from stochastic_proc.midprice import HistoricalMidprice
from stochastic_proc.arrivals import PoissonArrivals 

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
        assert self.max_depth is not None, "For limit orders max_depth cannot be None."
        # agent chooses spread on bid and ask
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

# stochastic_proc/dynamics.py (add this class)
class DOGEUSDTDynamics(LimitOrderDynamics):
    """
    Historical data-based dynamics specifically for DOGEUSDT L2 data
    """
    def __init__(self, feed, tick_size: float = 0.0001, quote_size: float = 100.0, 
                 max_depth: float = 0.01, num_traj: int = 1, 
                 fill_method: str = "cross", min_spread: float = 0.0001,
                 arrival_intensity: float = 1.0, sigma: float = 0.02):
        self.feed = feed
        self.tick_size = float(tick_size)
        self.quote_size = float(quote_size)
        self.max_depth = float(max_depth)
        self.num_traj = int(num_traj)
        self.fill_method = fill_method
        self.min_spread = min_spread
        
        # Create proper mid and arrival processes with volatility
        self.mid = HistoricalMidprice(feed, num_traj=num_traj, dt=1.0, T=10000, sigma=sigma)
        # self.arr = HistoricalArrivals(feed, arrival_intensity=arrival_intensity, num_traj=num_traj, dt=1.0, T=10000)
        
                # Initialize parent class
        super().__init__(self.mid, fill_k=1.0, max_depth=max_depth)
        
    def get_action_space(self):
        return gym.spaces.Box(low=0.0, high=self.max_depth, shape=(2,), dtype=np.float32)
    
    @property
    def midprice(self) -> np.ndarray:
        snapshot = self.feed.snapshot() if hasattr(self.feed, 'snapshot') else self.feed._get_batch_snapshot()
        mid = snapshot['mid']
        if np.isscalar(mid):
            return np.array([[mid]] * self.num_traj)
        return mid.reshape(-1, 1)
    
    def arrivals_and_fills(self, half_spreads: np.ndarray, rng):
        """Determine fills based on DOGEUSDT market data"""
        snapshot = self.feed.snapshot() if hasattr(self.feed, 'snapshot') else self.feed._get_batch_snapshot()
        
        if self.fill_method == "cross":
            return self._cross_fills(snapshot, half_spreads)
        elif self.fill_method == "improve":
            return self._improvement_fills(snapshot, half_spreads)
        else:  # probabilistic
            return self._probabilistic_fills(snapshot, half_spreads, rng)
    
    def _cross_fills(self, snapshot, half_spreads):
        """Fill if our quote crosses the best bid/ask"""
        best_bid = snapshot['best_bid']
        best_ask = snapshot['best_ask']
        mid = snapshot['mid']
        
        if np.isscalar(best_bid):
            best_bid = np.array([best_bid] * self.num_traj)
            best_ask = np.array([best_ask] * self.num_traj)
            mid = np.array([mid] * self.num_traj)
        
        # Our quotes
        our_bid = mid - half_spreads[:, 0]
        our_ask = mid + half_spreads[:, 1]
        
        # Fill if our bid >= best_ask (crosses spread) or our ask <= best_bid
        bid_fill = our_bid >= best_ask
        ask_fill = our_ask <= best_bid
        
        # Always consider arrivals in historical data
        arrivals = np.ones((self.num_traj, 2), dtype=bool)
        fills = np.column_stack([bid_fill, ask_fill])
        
        return arrivals, fills
    
    def _improvement_fills(self, snapshot, half_spreads):
        """Fill if our quote improves the best bid/ask"""
        best_bid = snapshot['best_bid']
        best_ask = snapshot['best_ask']
        mid = snapshot['mid']
        
        if np.isscalar(best_bid):
            best_bid = np.array([best_bid] * self.num_traj)
            best_ask = np.array([best_ask] * self.num_traj)
            mid = np.array([mid] * self.num_traj)
        
        # Our quotes
        our_bid = mid - half_spreads[:, 0]
        our_ask = mid + half_spreads[:, 1]
        
        # Fill if we improve the best bid/ask
        bid_fill = our_bid > best_bid
        ask_fill = our_ask < best_ask
        
        arrivals = np.ones((self.num_traj, 2), dtype=bool)
        fills = np.column_stack([bid_fill, ask_fill])
        
        return arrivals, fills
    
    def _probabilistic_fills(self, snapshot, half_spreads, rng):
        """Probabilistic fills based on distance from best quotes"""
        best_bid = snapshot['best_bid']
        best_ask = snapshot['best_ask']
        mid = snapshot['mid']
        
        if np.isscalar(best_bid):
            best_bid = np.array([best_bid] * self.num_traj)
            best_ask = np.array([best_ask] * self.num_traj)
            mid = np.array([mid] * self.num_traj)
        
        our_bid = mid - half_spreads[:, 0]
        our_ask = mid + half_spreads[:, 1]
        
        # Calculate distance from best quotes in ticks
        bid_improvement = (our_bid - best_bid) / self.tick_size
        ask_improvement = (best_ask - our_ask) / self.tick_size
        
        # Fill probability increases with improvement
        bid_fill_prob = 1.0 / (1.0 + np.exp(-2.0 * bid_improvement))
        ask_fill_prob = 1.0 / (1.0 + np.exp(-2.0 * ask_improvement))
        
        # Sample fills
        bid_fill = rng.random(self.num_traj) < bid_fill_prob
        ask_fill = rng.random(self.num_traj) < ask_fill_prob
        
        arrivals = np.ones((self.num_traj, 2), dtype=bool)
        fills = np.column_stack([bid_fill, ask_fill])
        
        return arrivals, fills
    
    def cash_inventory_delta(self, half_spreads: np.ndarray, arrivals: np.ndarray, fills: np.ndarray):
        """Calculate inventory and cash changes for DOGEUSDT"""
        snapshot = self.feed.snapshot() if hasattr(self.feed, 'snapshot') else self.feed._get_batch_snapshot()
        mid = snapshot['mid']
        
        if np.isscalar(mid):
            mid = np.array([mid] * self.num_traj)
        
        # Our execution prices
        bid_px = mid - half_spreads[:, 0]
        ask_px = mid + half_spreads[:, 1]
        
        # Only execute on fills
        executed_bid = fills[:, 0] & arrivals[:, 0]
        executed_ask = fills[:, 1] & arrivals[:, 1]
        
        dq = np.zeros(self.num_traj, dtype=int)
        dcash = np.zeros(self.num_traj, dtype=float)
        
        # Buy on bid executions, sell on ask executions
        dq[executed_bid] += 1
        dq[executed_ask] -= 1
        
        dcash[executed_bid] -= bid_px[executed_bid] * self.quote_size
        dcash[executed_ask] += ask_px[executed_ask] * self.quote_size
        
        return dq, dcash