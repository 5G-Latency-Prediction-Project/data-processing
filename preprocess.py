import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def ts_preprocess(data, window, normalize=False):
    '''
    This function is used to preprocess the time series data by denoising and 
    removing outliers. It also has other options like normalization.

    Parameters
    ----------
    data : pd.Series
        Time-series data.
    window : int
        The window size.
    normalize : Boolean, optional
        Normalizes the data for neural networks. The default is False.

    Returns
    -------
    Preprocessed time series data

    '''
    # Rolling mean denoising
    rm = data.rolling(window).mean()

    # Outlier detection
    roll = rm.rolling(window)
    avg = roll.mean()
    std = roll.std()
    outliers = (rm > avg+3*std) + (rm < avg-3*std)
    for oi in range(len(outliers)):
        if outliers[oi]:
            rm[oi] = avg[oi]
    
    if normalize:
        rm = (rm-avg)/std
    
    return rm

pq_filename = "C:/Users/ABRA/Desktop/5g_latency/s1/10-42-3-2_55500_20230809_114214.parquet"
df = pq.read_table(pq_filename).to_pandas()

# Calculate wall latency (timestamps.client.send.wall - timestamps.server.receive.wall)
wall_latency = pd.Series(data=(df['timestamps.server.receive.wall'] - df['timestamps.client.send.wall'])/1e6, index=df.index)
# Calculate monotonic latency (timestamps.client.send.monotonic - timestamps.server.receive.monotonic)
#monotonic_latency = pd.Series(data=(df['timestamps.server.receive.monotonic'] - df['timestamps.client.send.monotonic'])/1e6, index=df.index)

plt.plot(wall_latency)
plt.show()

processed = ts_preprocess(wall_latency, window=50)
plt.plot(processed)
plt.show()

norm = ts_preprocess(wall_latency, window=50, normalize=True)
plt.plot(norm)

