import math
import numpy as np
import matplotlib.pyplot as plt
import random

# Market parameters
S0 = 0.31  # Initial stock price
T = 1.0     # Trading period (1 day)
sigma = 3.5 # Volatility
M = 200     # Number of time steps
dt = T / M  # Time step size
Sim = 1000  # Number of simulations
gamma = 0.1 # Risk aversion
k = 0.18     # Order decay factor
A = 7    # Market order arrival rate

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
plt.plot(x,ReservPrice[:], lw = 1., label = 'r')
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
############################### Using Data ######################################

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

data = pd.read_csv("D:/Documents/CLS/thesis/MM_sandbox/binance_book_snapshot_25_2025-01-01_DOGEUSDT.csv", index_col = 'timestamp')
data.index = pd.to_datetime(data.index, unit='us')
data['midprice[0]'] = (data["asks[0].price"] + data["bids[0].price"]) / 2 

# Market parameters
S0 = 0.31  # Initial stock price
T = 1.0     # Trading period (1 day)
sigma = 2 # Volatility
M = len(data)     # Number of time steps
dt = T / M  # Time step size
Sim = 10  # Number of simulations
gamma = 0.1 # Risk aversion
k = 0.8     # Order decay factor
A = 7  # Market order arrival rate

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
S = data['midprice[0]'].to_numpy()
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
    for t in range(1, M):
        # Simulate price movement (Geometric Brownian Motion)
        # z = np.random.standard_normal()
        # S[t] = S[t-1] + sigma * math.sqrt(dt) * z  

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
# x = np.linspace(0., T, num= (M+1))
x = data.index
fig=plt.figure(figsize=(10,8))  
plt.subplot(2,1,1) # number of rows, number of  columns, number of the subplot 
plt.plot(x,S[:M], lw = 1., label = 'S')
plt.plot(x,ReservPrice[:M], lw = 1., label = 'r')
plt.plot(x,Ask[:M], lw = 1., label = 'r^a')
plt.plot(x,Bid[:M], lw = 1., label = 'r^b')       
plt.grid(True)
plt.legend(loc=0)
plt.ylabel('P')
plt.title('Prices')
plt.subplot(2,1,2)
plt.plot(x,q[:M], 'g', lw = 1., label = 'q') #plot 2 lines
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Inventory')

#Histogram of profit:
plt.figure(figsize = (7,5))
plt.hist(np.array(Profit), label = ['Inventory strategy'], bins = 10)
plt.legend(loc = 0)
plt.grid(True)
plt.xlabel('pnl')
plt.ylabel('number of values')
plt.title('Histogram')
    
# PnL path (single sim)
plt.figure(figsize=(7,5))
plt.plot(x, PnL[:M], label='Inventory strategy')
plt.legend(loc=0); plt.grid(True); plt.xlabel('Time'); plt.ylabel('PnL'); plt.title('Profit')
plt.show()

#################################################################################

import pandas as pd
import numpy as np
import math
import random
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style('darkgrid')

def resample_prices(ms, N):

    ms = int(ms)
    
    levels = pd.read_csv('D:/Documents/CLS/thesis/MM_sandbox/binance_book_snapshot_25_2025-01-01_DOGEUSDT.csv',index_col='timestamp')
    levels.index = pd.to_datetime(levels.index, unit="us")
    levels = levels[['asks[0].price', 'bids[0].price']]
    levels['mid'] = (levels['asks[0].price'] + levels['bids[0].price'])/2
    levels = levels.resample(f'{ms}ms').mean().ffill()
    
    logrets = np.log(levels['mid']).diff().dropna()
    eff_N = len(logrets)
    sigma = logrets.std() * np.sqrt(eff_N)

    return levels, levels.mid, sigma

clean_levels, s, sigma = resample_prices(1000,713815)

def create_arrays(N):

    pnl, x, q = np.empty((N)), np.empty((N)), np.empty((N))
    pnl[0], x[0], q[0] = 0, 0, 0
    o, r, ra, rb, rs = np.empty((N)), np.empty((N)), np.empty((N)), np.empty((N)), np.empty((N))
    db, da = np.empty((N)), np.empty((N))
    lb, la = np.empty((N)), np.empty((N))
    pb, pa = np.empty((N)), np.empty((N))
    f = np.empty((N))

    return pnl, x, q, o, r, rb, ra, rs, db, da, lb, la, pb, pa, f

def AS_sim(ms, gamma, A, k, reduce=False):

    SECONDS_PER_DAY = 60 * 60 * 24
    time_step = ms/1e3
    # N = int(SECONDS_PER_DAY/time_step)
    # T = 1
    # dt = 1/N

    clean_levels, s, sigma = resample_prices(ms, None)
    N = len(s)
    T = 1.0
    dt = 1.0 / N
    
    ds = s.diff().fillna(0)
    m = ds.shift(-5).rolling(5).mean().fillna(0)

    
    pnl, x, q, o, r, rb, ra, rs, db, da, lb, la, pb, pa, f  = create_arrays(N)

    for i in range(N-1):

        # Reserve price
        r[i] = s[i] - q[i] * gamma * sigma**2 * (T-dt*i)

        # Reserve spread
        rs[i] = gamma * sigma**2 * (T- dt*i) + 2 / gamma * math.log(1+gamma/k)    

        # optimal quotes
        ra[i] = r[i] + rs[i]/2
        rb[i] = r[i] - rs[i]/2

        # Cap our bid ask
        if ra[i] <= clean_levels["asks[0].price"][i]:
            ra[i] = clean_levels["asks[0].price"][i]
        
        if rb[i] >= clean_levels["bids[0].price"][i]:
            rb[i] = clean_levels["bids[0].price"][i]

        # Reserve deltas
        da[i] = ra[i] - s[i]
        db[i] = s[i] - rb[i]

        # Intensities
        lb[i] = A * math.exp(-k*db[i])
        la[i] = A * math.exp(-k*da[i])

        # Simulating probability of quotes getting hit/lifted
        yb = random.random()
        ya = random.random()

        pb[i] = (1 - math.exp(-lb[i]*(ms/1e3))) 
        pa[i] = (1 - math.exp(-la[i]*(ms/1e3))) 

        dNa, dNb = 0, 0

        if ya < pa[i]:
            dNa = 1
        if yb < pb[i]:
            dNb = 1
        
        f[i] = dNa + dNb
        q[i+1] = q[i] - dNa + dNb
        x[i+1] = x[i] + ra[i]*dNa - rb[i]*dNb
        pnl[i+1] = x[i+1] + q[i+1]*s[i]

    if reduce:
        return pnl[-1]

    data = {
        's': s,
        'ds': ds,
        'ms': m,
        'q': q,
        'o': o,
        'r': r,
        'rb': rb,
        'ra': ra,
        'rs': rs,
        'db': db,
        'da': da,
        #'b2': clean_levels.b2,
        'b1': clean_levels['bids[0].price'],
        # 'spr': clean_levels.a1 - clean_levels.b1,
        'a1': clean_levels['asks[0].price'],
        #'a2': clean_levels.a2,
        'lb': lb,
        'la': la,
        'pb': pb,
        'pa': pa,
        'pd': pa - pb,
        'pnl': pnl,
        'f': f.cumsum()
    }
    df = pd.DataFrame(data)
    df.set_index(s.index, inplace=True)
    df = df.iloc[:-1,:]

    return df

SECONDS_PER_DAY = 60 * 60 * 24
time_step = 100/1e3
N = int(SECONDS_PER_DAY/time_step) 
ms = 1e3

A, k = 7.46, 0.18
gamma = 0.1
np.random.seed(42)
df = AS_sim(ms, gamma, A, k)

df.pnl.plot(title='pnl')