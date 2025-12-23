###########################################################################
######################### AS-model with data ##############################
###########################################################################

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

rng = np.random.default_rng(123)

data = pd.read_csv("D:/Documents/CLS/thesis/MM_sandbox/binance_book_snapshot_25_2025-01-01_DOGEUSDT.csv", index_col = 'timestamp')
data.index = pd.to_datetime(data.index, unit='us')
data['midprice[0]'] = (data["asks[0].price"] + data["bids[0].price"]) / 2 
midprice = data['midprice[0]'].to_numpy()

# Market parameters
S0 = midprice[0]  # Initial stock price
T = 1.0     # Trading period (1 day)
sigma = 2 # Volatility
M = len(data)     # Number of time steps
dt = T / M  # Time step size
Sim = 10 # Number of simulations
gamma = 0.001 # Risk aversion
k = 4     # Order decay factor
A = 140  # Market order arrival rate

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
S = midprice
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

tick_size = 0.0001

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
        # Bid[t] = max(ReservPrice[t] - spread[t]/2., 0.0001)     
        # Ask[t] = max(ReservPrice[t] + spread[t]/2., 0.0001)
        
        Bid[t] = ReservPrice[t] - spread[t]/2.   
        Ask[t] = ReservPrice[t] + spread[t]/2.  

        # Compute order arrival probabilities (Poisson process)
        deltaB[t] = (S[t] - Bid[t]) / tick_size     
        deltaA[t] = (Ask[t] - S[t]) / tick_size

        lambdaA = A * math.exp(-k * deltaA[t])  # Arrival rate for ask orders # Estimate kappa (different for different stocks)
        ProbA = lambdaA * dt  # Probability of ask order execution
        fa = random.random() < ProbA # Random uniform sample for ask side
        
        lambdaB = A * math.exp(-k * deltaB[t])  # Arrival rate for bid orders
        ProbB = lambdaB * dt  # Probability of bid order execution
        fb = random.random() < ProbB # Random uniform sample for bid side

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
plt.plot(x,midprice, lw = 1., label = 'S')
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

###########################################################################
######################### AS-model Simulation #############################
###########################################################################

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