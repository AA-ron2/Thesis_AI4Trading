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
