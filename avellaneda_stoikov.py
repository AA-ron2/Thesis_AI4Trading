import math
import numpy as np
import matplotlib.pyplot as plt
import random

# Market parameters
S0 = 0.31  # Initial stock price
T = 1.0     # Trading period (1 day)
sigma = 2.0 # Volatility
M = 80000     # Number of time steps
dt = T / M  # Time step size
Sim = 1000  # Number of simulations
gamma = 0.1 # Risk aversion
k = 0.18     # Order decay factor
A = 7     # Market order arrival rate

# Initialize variables
S = np.zeros(M+1)
Bid = np.zeros(M+1)
Ask = np.zeros(M+1)
ReservPrice = np.zeros(M+1)
spread = np.zeros(M+1)
deltaB = np.zeros(M+1)
deltaA = np.zeros(M+1)
q = np.zeros(M+1)  # Inventory
w = np.zeros(M+1)  # Cash balance
PnL = np.zeros(M+1)  # Profit and Loss (PnL)

# Initial conditions
S[0] = S0
ReservPrice[0] = S0
Bid[0] = S0
Ask[0] = S0
spread[0] = 0
deltaB[0] = 0
deltaA[0] = 0
q[0] = 0  # Inventory (starts at 0)
w[0] = 0  # Wealth (starts at 0)
PnL[0] = 0

#Results:
AverageSpread = []
Profit = []
Std = []

for i in range(1, Sim+1):

    # Simulation loop
    for t in range(1, M+1):
        # Simulate price movement (Geometric Brownian Motion)
        z = np.random.standard_normal()
        S[t] = S[t-1] + sigma * math.sqrt(dt) * z  

        # Compute reservation price
        ReservPrice[t] = S[t] - q[t-1] * gamma * (sigma**2) * (T - t/M)

        # Compute optimal spread
        spread[t] = gamma * (sigma**2) * (T - t/M) + (2/gamma) * math.log(1 + (gamma/k))
        
        # Compute bid and ask prices
        Bid[t] = ReservPrice[t] - spread[t]/2.     
        Ask[t] = ReservPrice[t] + spread[t]/2.  

        # Compute order arrival probabilities (Poisson process)
        deltaB[t] = S[t] - Bid[t]     
        deltaA[t] = Ask[t] - S[t]

        lambdaA = A * math.exp(-k * deltaA[t])  # Arrival rate for ask orders # Estimate kappa (different for different stocks)
        ProbA = lambdaA * dt  # Probability of ask order execution
        fa = random.random()  # Random uniform sample for ask side
        
        lambdaB = A * math.exp(-k * deltaB[t])  # Arrival rate for bid orders
        ProbB = lambdaB * dt  # Probability of bid order execution
        fb = random.random()  # Random uniform sample for bid side

        # Execute trades based on probabilities
        if ProbB > fb and ProbA < fa:
            q[t] = q[t-1] + 1
            w[t] = w[t-1] - Bid[t]  # Buy at bid price

        elif ProbB < fb and ProbA > fa:
            q[t] = q[t-1] - 1
            w[t] = w[t-1] + Ask[t]  # Sell at ask price  

        elif ProbB < fb and ProbA < fa:
            q[t] = q[t-1]  # No trade
            w[t] = w[t-1]
        
        elif ProbB > fb and ProbA > fa:
            q[t] = q[t-1]  # Buy & sell at the same time (cancel out)
            w[t] = w[t-1] - Bid[t] + Ask[t]

        # Compute P&L
        PnL[t] = w[t] + q[t] * S[t]
        
    # Gather info
    AverageSpread.append(spread.mean())
    Profit.append(PnL[-1])
    Std.append(PnL[-1])
    
#Plots
x = np.linspace(0., T, num= (M+1))
    
fig=plt.figure(figsize=(10,8))  
plt.subplot(2,1,1) # number of rows, number of  columns, number of the subplot 
plt.plot(x,S[:], lw = 1., label = 'S')
plt.plot(x,Ask[:], lw = 1., label = 'r^a')
plt.plot(x,Bid[:], lw = 1., label = 'r^b')       
plt.grid(True)
plt.legend(loc=0)
plt.ylabel('P')
plt.title('Prices')
plt.subplot(2,1,2)
plt.plot(x,q[:], 'g', lw = 1., label = 'q') #plot 2 lines
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('Time')
plt.ylabel('Position')

#Histogram of profit:
plt.figure(figsize = (7,5))
plt.hist(np.array(Profit), label = ['Inventory strategy'], bins = 100)
plt.legend(loc = 0)
plt.grid(True)
plt.xlabel('pnl')
plt.ylabel('number of values')
plt.title('Histogram')
    
#PNL:
plt.figure(figsize = (7,5))
plt.plot(np.array(PnL), label = 'Inventory strategy')
plt.legend(loc = 0)
plt.grid(True)
plt.xlabel('Timesteps')
plt.ylabel('PnL')
plt.title('Profit')


#################################################################################

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ======================================================================
# 1) INPUTS FROM YOUR DATA
# ======================================================================
# If you already have a DataFrame 'df' loaded, keep it. Otherwise:

path = "D:/Documents/CLS/thesis/MM_sandbox/binance_book_snapshot_25_2025-01-01_DOGEUSDT.csv"
df = pd.read_csv(path)

BEST_ASK_COL = "asks[0].price"   # change if needed
BEST_BID_COL = "bids[0].price"   # change if needed
TS_COL       = "timestamp"            # change if needed (epoch timestamp)

# If your timestamps are NOT epoch (e.g., already seconds-since-start), set TS_UNIT='s'
TS_UNIT = "auto"  # one of {"auto","ns","us","ms","s"}

# ======================================================================
# 2) HELPER: build midprice & time grid from the DataFrame
# ======================================================================
def _infer_unit_from_epoch(ts: np.ndarray) -> str:
    # Infer unit from magnitude (epoch time in ~2025)
    vmax = float(np.max(np.abs(ts)))
    if vmax > 1e18: return "ns"   # nanoseconds since epoch
    if vmax > 1e15: return "us"   # microseconds since epoch
    if vmax > 1e12: return "ms"   # milliseconds since epoch
    if vmax > 1e9:  return "s"    # seconds since epoch
    # Otherwise treat as already seconds offset
    return "s"

def _to_seconds(ts: np.ndarray, unit: str) -> np.ndarray:
    if unit == "ns": scale = 1e9
    elif unit == "us": scale = 1e6
    elif unit == "ms": scale = 1e3
    elif unit == "s": scale = 1.0
    else:
        raise ValueError(f"Unsupported TS_UNIT={unit}")
    # If epoch-based, normalize to start at 0
    t0 = ts[0]
    return (ts - t0) / scale

def make_mid_and_time_from_df(df: pd.DataFrame, ask_col: str, bid_col: str, ts_col: str, ts_unit: str = "auto"):
    # Ensure sorted by timestamp
    df = df.sort_values(ts_col).reset_index(drop=True)
    ask = df[ask_col].to_numpy(dtype=float)
    bid = df[bid_col].to_numpy(dtype=float)
    mid = 0.5 * (ask + bid)

    ts_raw = df[ts_col].to_numpy()
    if ts_unit == "auto":
        unit = _infer_unit_from_epoch(ts_raw)
    else:
        unit = ts_unit
    t_sec = _to_seconds(ts_raw.astype(np.int64), unit)

    # Guard against non-increasing timestamps (drop duplicates)
    good = np.maximum.accumulate(t_sec) == t_sec
    if not np.all(good):
        t_sec = t_sec[good]
        mid   = mid[good]

    # Build per-step dt (seconds); set dt[0]=0 so cumulative time is correct
    dt = np.empty_like(t_sec)
    dt[0] = 0.0
    if len(t_sec) > 1:
        dt[1:] = np.diff(t_sec)

    T = float(t_sec[-1])                 # trading period from data (seconds)
    M = int(len(t_sec) - 1)              # number of steps
    return mid, t_sec, dt, T, M

# ======================================================================
# 3) MARKET/STRATEGY PARAMETERS (same as your script, keep/change as needed)
# ======================================================================
gamma = 0.1  # risk aversion
k     = 5 # order decay factor
A     = 140  # baseline arrival rate
Sim   = 10 # number of Monte Carlo simulations (random fills only)

# Volatility parameter for AS formulas (you can keep this,
# or compute realized sigma from mid if you prefer)
sigma = 2.0

# ======================================================================
# 4) BUILD THE PRICE & TIME GRID FROM DATA
# ======================================================================
# Use your already-loaded df here:
# df = ...
mid, t_sec, dt_series, T, M = make_mid_and_time_from_df(
    df, ask_col=BEST_ASK_COL, bid_col=BEST_BID_COL, ts_col=TS_COL, ts_unit=TS_UNIT
)

# Convenience arrays (length M+1)
S = mid.copy()      # midprice path from data
x = t_sec.copy()    # time axis in seconds

S0 = float(S[0])

# ======================================================================
# 5) SIMULATION OVER RANDOM FILLS (S is fixed from data)
# ======================================================================
AverageSpread = []
Profit = []

# We'll keep the last path to plot
last_Bid = np.zeros(M+1)
last_Ask = np.zeros(M+1)
last_q   = np.zeros(M+1)
last_PnL = np.zeros(M+1)

for i in range(Sim):
    Bid = np.zeros(M+1)
    Ask = np.zeros(M+1)
    ReservPrice = np.zeros(M+1)
    spread = np.zeros(M+1)
    deltaB = np.zeros(M+1)
    deltaA = np.zeros(M+1)
    q = np.zeros(M+1)     # inventory
    w = np.zeros(M+1)     # cash
    PnL = np.zeros(M+1)   # mark-to-market P&L

    # Initial conditions (use data mid)
    ReservPrice[0] = S0
    Bid[0] = S0
    Ask[0] = S0
    PnL[0] = 0.0
    q[0]   = 0.0
    w[0]   = 0.0

    for t in range(1, M+1):
        # Time to maturity (use actual remaining horizon from data)
        tau = max(0.0, T - x[t])  # seconds remaining

        # Reservation price & optimal spread (Avellaneda–Stoikov)
        ReservPrice[t] = S[t] - q[t-1] * gamma * (sigma**2) * tau
        spread[t] = gamma * (sigma**2) * tau + (2.0 / gamma) * math.log(1.0 + (gamma / k))

        # Quotes around reservation price
        Bid[t] = ReservPrice[t] - 0.5 * spread[t]
        Ask[t] = ReservPrice[t] + 0.5 * spread[t]

        # Distance to mid (in price units)
        deltaB[t] = S[t] - Bid[t]
        deltaA[t] = Ask[t] - S[t]

        # Arrival intensities and per-step execution probabilities
        lamA = A * math.exp(-k * deltaA[t])
        lamB = A * math.exp(-k * deltaB[t])

        dt = float(dt_series[t])   # seconds between t-1 and t (irregular OK)
        ProbA = max(0.0, lamA * dt)
        ProbB = max(0.0, lamB * dt)

        fa = random.random()
        fb = random.random()

        # Fills
        if ProbB > fb and ProbA < fa:
            # buy one at Bid
            q[t] = q[t-1] + 1.0
            w[t] = w[t-1] - Bid[t]
        elif ProbB < fb and ProbA > fa:
            # sell one at Ask
            q[t] = q[t-1] - 1.0
            w[t] = w[t-1] + Ask[t]
        elif ProbB > fb and ProbA > fa:
            # crossed: buy & sell in same step (inventory unchanged, capture spread)
            q[t] = q[t-1]
            w[t] = w[t-1] - Bid[t] + Ask[t]
        else:
            # no fill
            q[t] = q[t-1]
            w[t] = w[t-1]

        # Mark-to-market P&L on mid
        PnL[t] = w[t] + q[t] * S[t]

    AverageSpread.append(spread.mean())
    Profit.append(PnL[-1])

    # Keep last path for plotting
    if i == Sim - 1:
        last_Bid[:] = Bid
        last_Ask[:] = Ask
        last_q[:]   = q
        last_PnL[:] = PnL

# ======================================================================
# 6) PLOTS
# ======================================================================
fig = plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.plot(x, S,  lw=1., label='Mid (from data)')
plt.plot(x, last_Ask, lw=1., label='Ask quote')
plt.plot(x, last_Bid, lw=1., label='Bid quote')
plt.grid(True)
plt.legend(loc=0)
plt.ylabel('Price')
plt.title('Quotes on data-driven mid')

plt.subplot(2,1,2)
plt.plot(x, last_q, 'g', lw=1., label='Inventory q')
plt.grid(True)
plt.legend(loc=0)
plt.xlabel('Time (s)')
plt.ylabel('Position')

plt.figure(figsize=(7,5))
plt.hist(np.array(Profit), bins=100, label=['End P&L over sims'])
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('PnL')
plt.ylabel('Count')
plt.title('Histogram of end-of-period P&L')

plt.figure(figsize=(7,5))
plt.plot(x, last_PnL, label='PnL (last sim)')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('PnL')
plt.title('PnL path (last simulation)')
plt.show()