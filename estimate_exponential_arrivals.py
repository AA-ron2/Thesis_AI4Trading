#!/usr/bin/env python3
"""
Estimate exponential arrival parameters A, k from L2 order book data.

Event rule (Option A — best-quote crossover, snapshot-only):
- ASK side (buy limit at p0 = best_ask + δ*tick): event when best_bid >= p0
- BID side (sell limit at p0 = best_bid - δ*tick): event when best_ask <= p0
- Right-censor any pending wait at horizon T.

Outputs:
- estimated_rates.csv          (λ̂ by side and δ)
- estimated_fit_params.csv     (Â, k̂ per side, global and (optional) by windows)
- lambda_fit_ask.png / ...bid.png (if plotting enabled)

Usage examples:
  python estimate_exponential_arrivals.py --file /path/to/l2.csv
  python estimate_exponential_arrivals.py --file /path/to/l2.csv --deltas 0 1 2 3 4 5 6
  python estimate_exponential_arrivals.py --file /path/to/l2.csv --tick 0.01 --T 10
  python estimate_exponential_arrivals.py --file /path/to/l2.csv --by H   # per-hour windows
  python estimate_exponential_arrivals.py --file /path/to/l2.csv --no-plots

Notes:
- If --T is omitted, T = clip(0.1 * total_span, 0.5, 30) seconds.
- With *very* short samples you may see NaNs (not enough uncensored events).
"""

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_timestamp_column_name(columns: Iterable[str]) -> Optional[str]:
    for c in columns:
        if re.search(r"(time|timestamp|ts)", str(c), flags=re.I):
            return c
    return None


def parse_timestamp_series(s: pd.Series) -> pd.Series:
    """Parse a timestamp series to tz-naive pandas datetime64[ns]."""
    if pd.api.types.is_numeric_dtype(s):
        vmax = float(pd.to_numeric(s, errors="coerce").max())
        if vmax > 1e14:   # ns
            ts = pd.to_datetime(s.astype("int64"), unit="ns", utc=True)
        elif vmax > 1e12: # us
            ts = pd.to_datetime(s.astype("int64"), unit="us", utc=True)
        elif vmax > 1e10: # ms
            ts = pd.to_datetime(s.astype("int64"), unit="ms", utc=True)
        else:             # s
            ts = pd.to_datetime(s.astype("int64"), unit="s", utc=True)
    else:
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        if ts.isna().all():
            raise ValueError("Cannot parse timestamp column to datetime.")
    return ts.dt.tz_convert(None)


def sort_by_level(cols: List[str]) -> List[str]:
    def level(c):
        m = re.search(r"\[(\d+)\]", str(c))
        return int(m.group(1)) if m else 0
    return sorted(cols, key=level)


def infer_tick_from_prices(prices: np.ndarray) -> float:
    """Robust small-quantile grid step as tick guess (with light snapping)."""
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p)]
    if p.size < 2:
        return float("nan")
    diffs = np.diff(np.unique(np.round(np.sort(p), 12)))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("nan")
    tick = float(np.quantile(diffs, 0.05))
    # snap to {1,2,5} * 10^k if close
    for base in (1, 2, 5):
        for k in range(-10, 2):
            cand = base * (10 ** k)
            if cand > 0 and abs(tick - cand) / cand < 0.1:
                return round(cand, max(0, -k))
    return round(tick, 10)


def waiting_times_crossover(
    ts_ns: np.ndarray,
    best_bid: np.ndarray,
    best_ask: np.ndarray,
    deltas: Iterable[int],
    tick: float,
    T: float,
    side: str,
) -> Dict[int, List[Tuple[float, bool]]]:
    """
    Build waiting-time pairs (tau_sec, is_event) per delta.

    ts_ns: numpy datetime64[ns] array (sorted)
    best_bid, best_ask: float arrays aligned to ts_ns
    side: 'ask' or 'bid'
    """
    out = {d: [] for d in deltas}
    for d in deltas:
        active = False
        start = None  # datetime64[ns]
        p0 = None
        for i in range(len(ts_ns)):
            if not active:
                start = ts_ns[i]
                p0 = (best_ask[i] + d * tick) if side == "ask" else (best_bid[i] - d * tick)
                active = True
                continue

            # event check
            event = (best_bid[i] >= p0) if side == "ask" else (best_ask[i] <= p0)
            tau = (ts_ns[i] - start) / np.timedelta64(1, "s")
            tau = float(tau)

            if event:
                out[d].append((tau, True))
                active = False
                continue
            if tau >= T:
                out[d].append((T, False))
                active = False
                continue

        if active:
            tau_end = (ts_ns[-1] - start) / np.timedelta64(1, "s")
            out[d].append((float(min(T, tau_end)), False))
    return out


def censored_exp_rate(pairs: List[Tuple[float, bool]], T: float) -> Tuple[float, int, float]:
    if not pairs:
        return float("nan"), 0, 0.0
    num = sum(1 for (_, ev) in pairs if ev)
    den = sum(min(t, T) for (t, _) in pairs)
    lam = (num / den) if den > 0 else float("nan")
    return lam, num, float(den)


def rates_for_side(wt: Dict[int, List[Tuple[float, bool]]], side: str, T: float) -> pd.DataFrame:
    rows = []
    for d in sorted(wt.keys()):
        lam, n, exp_time = censored_exp_rate(wt[d], T)
        rows.append(dict(side=side, delta=int(d), lambda_hat=lam, uncensored=n, exposure_seconds=exp_time))
    return pd.DataFrame(rows)


def fit_log_linear(rates: pd.DataFrame, side: str) -> Tuple[float, float, int]:
    r = rates[(rates["side"] == side) & np.isfinite(rates["lambda_hat"]) & (rates["lambda_hat"] > 0)]
    if r.empty or r["delta"].nunique() < 2:
        return float("nan"), float("nan"), 0
    D = r["delta"].to_numpy(dtype=float)
    y = np.log(r["lambda_hat"].to_numpy(dtype=float))
    X = np.c_[np.ones_like(D), -D]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    c, k = beta[0], beta[1]
    A = float(np.exp(c))
    return A, float(k), int(len(D))


def plot_rates_and_fit(rates: pd.DataFrame, A: float, k: float, side: str, out_path: Path):
    r = rates[(rates["side"] == side) & np.isfinite(rates["lambda_hat"]) & (rates["lambda_hat"] > 0)]
    plt.figure()
    plt.scatter(r["delta"], r["lambda_hat"])
    if math.isfinite(A) and math.isfinite(k) and not r.empty:
        xs = np.linspace(r["delta"].min(), r["delta"].max(), 100)
        ys = A * np.exp(-k * xs)
        plt.plot(xs, ys)
        title = f"{side.upper()} side: λ̂(δ) with fit (A={A:.4g}, k={k:.4g})"
    else:
        title = f"{side.upper()} side: λ̂(δ) (not enough points to fit)"
    plt.title(title)
    plt.xlabel("delta (ticks)")
    plt.ylabel("lambda_hat (1/sec)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def auto_horizon_T(ts: pd.Series) -> float:
    total_secs = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    if total_secs <= 0:
        total_secs = 1.0
    return max(0.5, min(30.0, 0.1 * total_secs))


def load_minimal_df(path: Path) -> pd.DataFrame:
    """Read only timestamp + ask/bid price columns to save memory."""
    # peek header
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    ts_col = find_timestamp_column_name(cols)
    if ts_col is None:
        raise ValueError("No timestamp-like column found (e.g., 'timestamp', 'ts').")

    ask_cols = [c for c in cols if re.fullmatch(r"asks\[\d+\]\.price", str(c))]
    bid_cols = [c for c in cols if re.fullmatch(r"bids\[\d+\]\.price", str(c))]
    if not ask_cols or not bid_cols:
        # loose match as fallback
        ask_cols = [c for c in cols if re.search(r"asks?\[\d+\]\.price", str(c))]
        bid_cols = [c for c in cols if re.search(r"bids?\[\d+\]\.price", str(c))]
    if not ask_cols or not bid_cols:
        raise ValueError("Could not find L2 price columns like 'asks[0].price' and 'bids[0].price'.")

    usecols = [ts_col] + ask_cols + bid_cols
    df = pd.read_csv(path, usecols=usecols)
    ts = parse_timestamp_series(df[ts_col])
    df = df.assign(__ts=ts).drop(columns=[ts_col]).sort_values("__ts").reset_index(drop=True)

    ask_cols = sort_by_level(ask_cols)
    bid_cols = sort_by_level(bid_cols)

    # best quotes
    best_ask = df[ask_cols].min(axis=1).astype(float).to_numpy()
    best_bid = df[bid_cols].max(axis=1).astype(float).to_numpy()

    return pd.DataFrame({"__ts": df["__ts"], "best_ask": best_ask, "best_bid": best_bid})


def estimate_once(
    core: pd.DataFrame,
    deltas: List[int],
    T: float,
    tick: Optional[float],
    make_plots: bool,
    outdir: Path,
    label_suffix: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run estimation on a core dataframe with columns __ts, best_ask, best_bid."""
    # infer tick if needed
    if tick is None or not np.isfinite(tick) or tick <= 0:
        tick = infer_tick_from_prices(np.concatenate([core["best_ask"].to_numpy(), core["best_bid"].to_numpy()]))

    # build arrays
    ts_ns = core["__ts"].to_numpy(dtype="datetime64[ns]")
    best_ask = core["best_ask"].to_numpy(dtype=float)
    best_bid = core["best_bid"].to_numpy(dtype=float)

    # waiting times per side
    wt_ask = waiting_times_crossover(ts_ns, best_bid, best_ask, deltas, tick, T, side="ask")
    wt_bid = waiting_times_crossover(ts_ns, best_bid, best_ask, deltas, tick, T, side="bid")

    # rates
    rates_ask = rates_for_side(wt_ask, "ask", T)
    rates_bid = rates_for_side(wt_bid, "bid", T)
    rates = pd.concat([rates_ask, rates_bid], ignore_index=True)
    rates["tick"] = tick
    if label_suffix:
        rates["window"] = label_suffix

    # fits
    A_ask, k_ask, n_ask = fit_log_linear(rates, "ask")
    A_bid, k_bid, n_bid = fit_log_linear(rates, "bid")
    fit = pd.DataFrame(
        [
            dict(side="ask", A_hat=A_ask, k_hat=k_ask, used_points=n_ask, tick=tick, window=label_suffix),
            dict(side="bid", A_hat=A_bid, k_hat=k_bid, used_points=n_bid, tick=tick, window=label_suffix),
        ]
    )

    # plots
    if make_plots:
        plot_rates_and_fit(rates, A_ask, k_ask, "ask", outdir / f"lambda_fit_ask{label_suffix and '_'+label_suffix}.png")
        plot_rates_and_fit(rates, A_bid, k_bid, "bid", outdir / f"lambda_fit_bid{label_suffix and '_'+label_suffix}.png")

    return rates, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, type=Path, help="Path to CSV with L2 data")
    ap.add_argument("--deltas", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Delta values (in ticks)")
    ap.add_argument("--tick", type=float, default=None, help="Override tick size (if known)")
    ap.add_argument("--T", type=float, default=None, help="Right-censoring horizon in seconds (default auto)")
    ap.add_argument("--by", type=str, default=None, help="Optional time window frequency, e.g. 'H' or '15min'")
    ap.add_argument("--outdir", type=Path, default=Path("./out"), help="Output directory")
    ap.add_argument("--no-plots", action="store_true", help="Disable plotting")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    core = load_minimal_df(args.file)
    core = core.sort_values("__ts").reset_index(drop=True)

    # choose T
    T = args.T if args.T and args.T > 0 else auto_horizon_T(core["__ts"])

    # global run
    rates_all, fit_all = estimate_once(
        core=core,
        deltas=args.deltas,
        T=T,
        tick=args.tick,
        make_plots=not args.no_plots,
        outdir=args.outdir,
        label_suffix="global",
    )

    # optional time-windowed runs
    fit_windows = []
    rates_windows = []
    if args.by:
        core_idx = core.set_index("__ts")
        for win_start, grp in core_idx.groupby(pd.Grouper(freq=args.by)):
            if grp.shape[0] < 3:
                continue
            label = win_start.strftime("%Y%m%dT%H%M%S")
            r_w, f_w = estimate_once(
                core=grp.reset_index(),
                deltas=args.deltas,
                T=T,           # keep same T for comparability; or recompute auto_horizon_T(grp["__ts"])
                tick=args.tick,
                make_plots=False,   # avoid making tons of PNGs
                outdir=args.outdir,
                label_suffix=label,
            )
            rates_windows.append(r_w)
            fit_windows.append(f_w)

    # write outputs
    rates_out = args.outdir / "estimated_rates.csv"
    fit_out = args.outdir / "estimated_fit_params.csv"

    rates_all.to_csv(rates_out, index=False, mode="w")
    fit_all.to_csv(fit_out, index=False, mode="w")

    if rates_windows:
        pd.concat(rates_windows, ignore_index=True).to_csv(args.outdir / "estimated_rates_by_window.csv", index=False)
    if fit_windows:
        pd.concat(fit_windows, ignore_index=True).to_csv(args.outdir / "estimated_fit_params_by_window.csv", index=False)

    # console summary
    print(f"Saved: {rates_out}")
    print(f"Saved: {fit_out}")
    if not args.no_plots:
        print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()
