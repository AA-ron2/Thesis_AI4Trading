#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------- Helpers ---------
def find_timestamp_column_name(columns: Iterable[str]) -> Optional[str]:
    for c in columns:
        if re.search(r"(time|timestamp|ts)", str(c), flags=re.I):
            return c
    return None

def parse_timestamp_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        vmax = float(pd.to_numeric(s, errors="coerce").max())
        if vmax > 1e14:
            ts = pd.to_datetime(s.astype("int64"), unit="ns", utc=True)
        elif vmax > 1e12:
            ts = pd.to_datetime(s.astype("int64"), unit="us", utc=True)
        elif vmax > 1e10:
            ts = pd.to_datetime(s.astype("int64"), unit="ms", utc=True)
        else:
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
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p)]
    if p.size < 2:
        return float("nan")
    diffs = np.diff(np.unique(np.round(np.sort(p), 12)))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("nan")
    tick = float(np.quantile(diffs, 0.05))
    for base in (1, 2, 5):
        for k in range(-10, 2):
            cand = base * (10 ** k)
            if cand > 0 and abs(tick - cand) / cand < 0.1:
                return round(cand, max(0, -k))
    return round(tick, 10)

# --------- A,k estimation (crossover wait-times) ---------
def waiting_times_crossover(ts_ns, best_bid, best_ask, deltas, tick, T, side):
    out = {d: [] for d in deltas}
    for d in deltas:
        active = False
        start = None
        p0 = None
        for i in range(len(ts_ns)):
            if not active:
                start = ts_ns[i]
                p0 = (best_ask[i] + d * tick) if side == "ask" else (best_bid[i] - d * tick)
                active = True
                continue
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

def censored_exp_rate(pairs, T):
    if not pairs:
        return float("nan"), 0, 0.0
    num = sum(1 for (_, ev) in pairs if ev)
    den = sum(min(t, T) for (t, _) in pairs)
    lam = (num / den) if den > 0 else float("nan")
    return lam, num, float(den)

def rates_for_side(wt, side, T):
    rows = []
    for d in sorted(wt.keys()):
        lam, n, exp_time = censored_exp_rate(wt[d], T)
        rows.append(dict(side=side, delta=int(d), lambda_hat=lam, uncensored=n, exposure_seconds=exp_time))
    return pd.DataFrame(rows)

def fit_log_linear(rates, side):
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

# --------- Data loading ---------
def load_minimal_df(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    ts_col = find_timestamp_column_name(cols)
    if ts_col is None:
        raise ValueError("No timestamp-like column found (e.g., 'timestamp', 'ts').")
    ask_cols = [c for c in cols if re.fullmatch(r"asks\[\d+\]\.price", str(c))]
    bid_cols = [c for c in cols if re.fullmatch(r"bids\[\d+\]\.price", str(c))]
    if not ask_cols or not bid_cols:
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
    best_ask = df[ask_cols].min(axis=1).astype(float).to_numpy()
    best_bid = df[bid_cols].max(axis=1).astype(float).to_numpy()
    return pd.DataFrame({"__ts": df["__ts"], "best_ask": best_ask, "best_bid": best_bid})

def auto_horizon_T(ts: pd.Series) -> float:
    total_secs = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    if total_secs <= 0:
        total_secs = 1.0
    return max(0.5, min(30.0, 0.1 * total_secs))

# --------- Sigma estimation ---------
def estimate_sigma(mid: pd.Series, ts: pd.Series, dt_sec: float = 1.0) -> float:
    df = pd.DataFrame({"mid": mid.to_numpy()}, index=ts)
    df = df[~df["mid"].isna()]
    resampled = df.resample(f"{dt_sec}S").last().ffill()
    log_mid = np.log(resampled["mid"].to_numpy())
    r = np.diff(log_mid)
    if r.size < 2:
        return float("nan")
    var = np.var(r, ddof=1) / dt_sec
    sigma = math.sqrt(var)
    return sigma

# --------- Plotting ---------
def plot_lambda_fit(rates: pd.DataFrame, A: float, k: float, side: str, out_path: Path):
    r = rates[(rates["side"] == side) & np.isfinite(rates["lambda_hat"]) & (rates["lambda_hat"] > 0)]
    if r.empty:
        return
    plt.figure(figsize=(5, 3.5))
    plt.plot(
        r["delta"], r["lambda_hat"],
        linestyle="--", linewidth=1.6,
        marker="o", markersize=5, markerfacecolor="none",
        markeredgewidth=1.2, color="tab:blue", label="lambda_hat"
    )
    if math.isfinite(A) and math.isfinite(k):
        D_line = np.linspace(r["delta"].min(), r["delta"].max(), 300)
        y_line = A * np.exp(-k * D_line)
        plt.plot(D_line, y_line, color="red", linewidth=2.0, label="regression")
    plt.title(f"λ̂ and fit ({side})")
    plt.xlabel("δ (ticks)")
    plt.ylabel("λ (events/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_lambda_ticks(rates: pd.DataFrame, A_ask: float, k_ask: float, A_bid: float, k_bid: float, out_path: Path):
    plt.figure(figsize=(5.5, 3.5))
    r_ask = rates[(rates["side"] == "ask") & np.isfinite(rates["lambda_hat"]) & (rates["lambda_hat"] > 0)]
    r_bid = rates[(rates["side"] == "bid") & np.isfinite(rates["lambda_hat"]) & (rates["lambda_hat"] > 0)]

    if not r_ask.empty:
        plt.plot(r_ask["delta"], r_ask["lambda_hat"],
                 linestyle="--", marker="o", markersize=5, markerfacecolor="none",
                 markeredgewidth=1.2, color="tab:red", label="ask λ̂")
        if math.isfinite(A_ask) and math.isfinite(k_ask):
            D_line = np.linspace(r_ask["delta"].min(), r_ask["delta"].max(), 300)
            plt.plot(D_line, A_ask * np.exp(-k_ask * D_line), color="tab:red", linewidth=2.0, label="ask fit")

    if not r_bid.empty:
        plt.plot(r_bid["delta"], r_bid["lambda_hat"],
                 linestyle="--", marker="s", markersize=5, markerfacecolor="none",
                 markeredgewidth=1.2, color="tab:blue", label="bid λ̂")
        if math.isfinite(A_bid) and math.isfinite(k_bid):
            D_line = np.linspace(r_bid["delta"].min(), r_bid["delta"].max(), 300)
            plt.plot(D_line, A_bid * np.exp(-k_bid * D_line), color="tab:blue", linewidth=2.0, label="bid fit")

    plt.title("λ̂ vs δ (ticks)")
    plt.xlabel("δ (ticks)")
    plt.ylabel("λ (events/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_mid_price(ts: pd.Series, mid: pd.Series, out_path: Path):
    plt.figure(figsize=(6, 3.0))
    plt.plot(ts, mid, color="tab:green", linewidth=1.0)
    plt.title("Mid price")
    plt.xlabel("time")
    plt.ylabel("price")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# --------- Glue ---------
def estimate_all(
    path: Path,
    deltas: List[int],
    tick_override: Optional[float],
    T_override: Optional[float],
    sigma_dt: float,
    outdir: Path,
    make_plots: bool,
):
    core = load_minimal_df(path)
    core = core.sort_values("__ts").reset_index(drop=True)
    tick = tick_override
    if tick is None or not np.isfinite(tick) or tick <= 0:
        tick = infer_tick_from_prices(np.concatenate([core["best_ask"].to_numpy(), core["best_bid"].to_numpy()]))
    T = T_override if T_override and T_override > 0 else auto_horizon_T(core["__ts"])
    ts_ns = core["__ts"].to_numpy(dtype="datetime64[ns]")
    best_ask = core["best_ask"].to_numpy(dtype=float)
    best_bid = core["best_bid"].to_numpy(dtype=float)

    wt_ask = waiting_times_crossover(ts_ns, best_bid, best_ask, deltas, tick, T, side="ask")
    wt_bid = waiting_times_crossover(ts_ns, best_bid, best_ask, deltas, tick, T, side="bid")
    rates_ask = rates_for_side(wt_ask, "ask", T)
    rates_bid = rates_for_side(wt_bid, "bid", T)
    rates = pd.concat([rates_ask, rates_bid], ignore_index=True)
    rates["tick"] = tick

    A_ask, k_ask, n_ask = fit_log_linear(rates, "ask")
    A_bid, k_bid, n_bid = fit_log_linear(rates, "bid")

    mid = (core["best_ask"] + core["best_bid"]) / 2
    sigma = estimate_sigma(mid, core["__ts"], dt_sec=sigma_dt)

    if make_plots:
        outdir.mkdir(parents=True, exist_ok=True)
        plot_lambda_fit(rates, A_ask, k_ask, "ask", outdir / "lambda_fit_ask.png")
        plot_lambda_fit(rates, A_bid, k_bid, "bid", outdir / "lambda_fit_bid.png")
        plot_lambda_ticks(rates, A_ask, k_ask, A_bid, k_bid, outdir / "lambda_ticks_combined.png")
        plot_mid_price(core["__ts"], mid, outdir / "mid_price.png")

    return dict(
        tick=tick,
        T=T,
        sigma_per_s=sigma,
        A_ask=A_ask,
        k_ask=k_ask,
        used_points_ask=n_ask,
        A_bid=A_bid,
        k_bid=k_bid,
        used_points_bid=n_bid,
        rates=rates,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, type=Path, help="Path to L2 CSV")
    ap.add_argument("--deltas", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Delta values (in ticks)")
    ap.add_argument("--tick", type=float, default=None, help="Override tick size (if known)")
    ap.add_argument("--T", type=float, default=None, help="Right-censor horizon in seconds (default auto)")
    ap.add_argument("--sigma-dt", type=float, default=1.0, help="Resampling step (s) for sigma")
    ap.add_argument("--outdir", type=Path, default=Path("./out"), help="Where to save plots")
    ap.add_argument("--no-plots", action="store_true", help="Disable plots")
    args = ap.parse_args()

    res = estimate_all(
        path=args.file,
        deltas=args.deltas,
        tick_override=args.tick,
        T_override=args.T,
        sigma_dt=args.sigma_dt,
        outdir=args.outdir,
        make_plots=not args.no_plots,
    )

    print(f"Inferred tick: {res['tick']}")
    print(f"Horizon T (s): {res['T']}")
    print(f"Sigma (per sqrt(sec)): {res['sigma_per_s']}")
    print("--- A,k fits ---")
    print(f"Ask: A={res['A_ask']}, k={res['k_ask']}, used_points={res['used_points_ask']}")
    print(f"Bid: A={res['A_bid']}, k={res['k_bid']}, used_points={res['used_points_bid']}")
    print("\nRates (lambda_hat per delta):")
    print(res["rates"])

if __name__ == "__main__":
    main()
