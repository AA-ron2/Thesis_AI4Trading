# data/feed.py
import numpy as np

class L2Feed:
    """
    Zero-copy replay of preprocessed arrays (could be numpy.memmap).
    Expect arrays of shape (T,) for prices and (T, L) for sizes.
    """
    def __init__(self, *, 
                 best_bid_px, best_ask_px,
                 best_bid_sz=None, best_ask_sz=None,  # optional (T, L)
                 trade_px=None, trade_sz=None, trade_side=None,  # optional
                 ts=None, idx_start=0, idx_end=None):
        self.best_bid_px = best_bid_px
        self.best_ask_px = best_ask_px
        self.best_bid_sz = best_bid_sz
        self.best_ask_sz = best_ask_sz
        self.trade_px = trade_px
        self.trade_sz = trade_sz
        self.trade_side = trade_side  # +1 buy MO, -1 sell MO (if available)
        self.ts = ts
        self.i0 = int(idx_start)
        self.i1 = len(best_bid_px) if idx_end is None else int(idx_end)
        self.T = self.i1 - self.i0
        self.t = 0

    def reset(self, offset=0):
        self.t = int(offset)
        return self.snapshot()

    def done(self):
        return self.t >= self.T - 1

    def step(self):
        self.t += 1
        return self.snapshot()

    def snapshot(self):
        i = self.i0 + self.t
        bb = self.best_bid_px[i]
        ba = self.best_ask_px[i]
        mid = 0.5 * (bb + ba)
        snap = dict(
            mid=mid, best_bid=bb, best_ask=ba,
            ts=(self.ts[i] if self.ts is not None else self.t)
        )
        if self.best_bid_sz is not None:
            snap["bid_sizes"] = self.best_bid_sz[i]  # (L,)
        if self.best_ask_sz is not None:
            snap["ask_sizes"] = self.best_ask_sz[i]  # (L,)
        if self.trade_px is not None:
            snap["trade_px"] = self.trade_px[i]
            snap["trade_sz"] = self.trade_sz[i]
            snap["trade_side"] = self.trade_side[i]
        return snap
