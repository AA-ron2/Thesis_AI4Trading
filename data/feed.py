# Re-define the preprocessing function and re-run it
import os
import pandas as pd

# Define the preprocessing function again
def preprocess_l2_data(file_path: str):
    """
    Preprocess the L2 order book data from the given file path.
    1. Loads the data.
    2. Preprocesses it (timestamp conversion, cleaning, restructuring).
    3. Returns a preprocessed DataFrame and a data loader for RL integration.
    """
    # Load the L2 order book data from CSV
    l2_data = pd.read_csv(file_path)
    
    # Convert timestamps to datetime (in microseconds)
    l2_data['timestamp'] = pd.to_datetime(l2_data['timestamp'], unit='us')
    
    # Clean the data (remove rows with missing values)
    l2_data_cleaned = l2_data.dropna()
    
    # Extract relevant columns (ask and bid prices and amounts for each level)
    ask_columns = [f'asks[{i}].price' for i in range(25)] + [f'asks[{i}].amount' for i in range(25)]
    bid_columns = [f'bids[{i}].price' for i in range(25)] + [f'bids[{i}].amount' for i in range(25)]
    l2_data_restructured = l2_data_cleaned[['timestamp'] + ask_columns + bid_columns]
    
    # Create a data loader for sequential access to the order book data
    class L2OrderBookLoader:
        def __init__(self, data: pd.DataFrame):
            self.data = data
            self.index = 0
            self.max_index = len(data) - 1

        def reset(self):
            self.index = 0

        def step(self):
            # Return the current snapshot (ask/bid prices, sizes, and timestamp)
            snapshot = self.data.iloc[self.index]
            # Increment the index
            self.index += 1
            if self.index > self.max_index:
                return None  # End of data
            # Return the snapshot with relevant data
            return {
                'timestamp': snapshot['timestamp'],
                'asks': snapshot[ask_columns].to_numpy(),
                'bids': snapshot[bid_columns].to_numpy()
            }
    
    # Return the preprocessed data and the loader
    l2_loader = L2OrderBookLoader(l2_data_restructured)
    return l2_data_restructured, l2_loader

# Re-run the preprocessing function on the new data
file_path = os.path.join("D:/Documents/CLS/thesis/MM_sandbox", 'binance_book_snapshot_25_2025-01-01_DOGEUSDT.csv')
preprocessed_data, data_loader = preprocess_l2_data(file_path)

# Show the first few rows of the preprocessed data
preprocessed_data.head()


# data/feed.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import re

class DOGEUSDTL2Feed:
    """
    Feed for DOGEUSDT L2 data with the specific binance format
    """
    def __init__(self, data: pd.DataFrame, num_traj: int = 1):
        """
        Args:
            data: DataFrame with L2 data in the exact format you provided
            num_traj: Number of parallel trajectories
        """
        self.data = data.reset_index(drop=True)
        self.num_traj = num_traj
        self.current_idx = 0
        self.max_idx = len(data) - 1
        
        # Pre-process the data to extract levels efficiently
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Pre-process data to extract bid/ask levels efficiently"""
        # Extract all bid and ask levels
        self.bid_prices = []
        self.bid_sizes = []
        self.ask_prices = []
        self.ask_sizes = []
        self.mid_prices = []
        self.best_bids = []
        self.best_asks = []
        
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            
            # Extract bids and asks using the array format
            bid_prices_row = []
            bid_sizes_row = []
            ask_prices_row = []
            ask_sizes_row = []
            
            # Parse the array format: bids[0].price, bids[0].amount, etc.
            for i in range(25):  # 25 levels
                bid_price_col = f'bids[{i}].price'
                bid_size_col = f'bids[{i}].amount'
                ask_price_col = f'asks[{i}].price'
                ask_size_col = f'asks[{i}].amount'
                
                if bid_price_col in row:
                    bid_prices_row.append(float(row[bid_price_col]))
                    bid_sizes_row.append(float(row[bid_size_col]))
                    ask_prices_row.append(float(row[ask_price_col]))
                    ask_sizes_row.append(float(row[ask_size_col]))
            
            self.bid_prices.append(bid_prices_row)
            self.bid_sizes.append(bid_sizes_row)
            self.ask_prices.append(ask_prices_row)
            self.ask_sizes.append(ask_sizes_row)
            
            # Calculate mid price and best bid/ask
            if bid_prices_row and ask_prices_row:
                best_bid = bid_prices_row[0]
                best_ask = ask_prices_row[0]
                self.best_bids.append(best_bid)
                self.best_asks.append(best_ask)
                self.mid_prices.append((best_bid + best_ask) / 2)
            else:
                self.best_bids.append(np.nan)
                self.best_asks.append(np.nan)
                self.mid_prices.append(np.nan)
        
        # Convert to numpy arrays for efficiency
        self.bid_prices = np.array(self.bid_prices)
        self.bid_sizes = np.array(self.bid_sizes)
        self.ask_prices = np.array(self.ask_prices)
        self.ask_sizes = np.array(self.ask_sizes)
        self.mid_prices = np.array(self.mid_prices)
        self.best_bids = np.array(self.best_bids)
        self.best_asks = np.array(self.best_asks)
    
    def reset(self, start_idx: int = 0) -> Dict[str, Any]:
        """Reset feed to specific index"""
        self.current_idx = start_idx
        return self.snapshot()
    
    def step(self) -> Dict[str, Any]:
        """Advance to next data point"""
        if self.current_idx < self.max_idx:
            self.current_idx += 1
        return self.snapshot()
    
    def snapshot(self) -> Dict[str, Any]:
        """Get current market snapshot"""
        if self.current_idx > self.max_idx:
            raise IndexError("Data feed index out of bounds")
            
        snapshot = {
            'timestamp': self.data.iloc[self.current_idx]['timestamp'],
            'mid': float(self.mid_prices[self.current_idx]),
            'best_bid': float(self.best_bids[self.current_idx]),
            'best_ask': float(self.best_asks[self.current_idx]),
            'bid_prices': self.bid_prices[self.current_idx].tolist(),
            'bid_sizes': self.bid_sizes[self.current_idx].tolist(),
            'ask_prices': self.ask_prices[self.current_idx].tolist(),
            'ask_sizes': self.ask_sizes[self.current_idx].tolist(),
            'current_idx': self.current_idx
        }
        
        return snapshot
    
    @property
    def is_done(self) -> bool:
        return self.current_idx >= self.max_idx
    
    def get_time_index(self) -> int:
        return self.current_idx

class BatchDOGEUSDTFeed:
    """
    Batch feed for multiple parallel trajectories through DOGEUSDT data
    """
    def __init__(self, data: pd.DataFrame, num_traj: int, episode_length: int):
        self.data = data
        self.num_traj = num_traj
        self.episode_length = episode_length
        self.max_start_idx = len(data) - episode_length - 1
        
        # Pre-process data
        self._preprocess_data()
        
        self.current_indices = None
        self.current_step = 0
        
    def _preprocess_data(self):
        """Pre-process the entire dataset for efficient batch operations"""
        self.mid_prices = []
        self.best_bids = []
        self.best_asks = []
        
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            
            # Extract best bid/ask from level 0
            best_bid = float(row['bids[0].price'])
            best_ask = float(row['asks[0].price'])
            
            self.best_bids.append(best_bid)
            self.best_asks.append(best_ask)
            self.mid_prices.append((best_bid + best_ask) / 2)
        
        # Convert to numpy arrays
        self.mid_prices = np.array(self.mid_prices)
        self.best_bids = np.array(self.best_bids)
        self.best_asks = np.array(self.best_asks)
    
    def reset(self) -> Dict[str, Any]:
        """Reset all trajectories to random starting points"""
        # Sample random start indices for each trajectory
        self.current_indices = np.random.randint(0, self.max_start_idx, self.num_traj)
        self.current_step = 0
        return self._get_batch_snapshot()
    
    def step(self) -> Dict[str, Any]:
        """Advance all trajectories by one step"""
        if self.current_step >= self.episode_length:
            raise ValueError("Episode completed, call reset() first")
            
        self.current_indices += 1
        self.current_step += 1
        return self._get_batch_snapshot()
    
    def _get_batch_snapshot(self) -> Dict[str, Any]:
        """Get batch snapshot for all trajectories"""
        batch_snapshot = {
            'mid': self.mid_prices[self.current_indices],
            'best_bid': self.best_bids[self.current_indices],
            'best_ask': self.best_asks[self.current_indices],
            'current_indices': self.current_indices.copy()
        }
        
        return batch_snapshot
    
    @property
    def is_done(self) -> bool:
        return self.current_step >= self.episode_length