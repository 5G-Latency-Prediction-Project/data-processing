import pandas as pd
import numpy as np
from scipy import signal

def ts_preprocess(data, window, diff=False, normalize=False, win_type=None):
    '''
    This function is used to preprocess the time series data by denoising and 
    removing outliers. It also has other options like normalization.

    Parameters
    ----------
    data : pd.Series
        Time-series data.
    window : int
        The window size.
    diff : Boolean, optional
        Performs a first order difference to remove non-stationarity. The default is False.
    normalize : Boolean, optional
        Normalizes the data for neural networks. The default is False.

    Returns
    -------
    Preprocessed time series data

    '''
    # Rolling mean denoising
    rm = data.rolling(window, win_type=win_type).mean()

    # Outlier detection
    roll = rm.rolling(window)
    avg = roll.mean()
    std = roll.std()
    outliers = (rm > avg+3*std) + (rm < avg-3*std)
    for oi in range(len(outliers)):
        if outliers[oi]:
            rm[oi] = avg[oi]
    
    if diff:
        rm = rm.diff()
    
    if normalize:
        rm = (rm-avg)/std
    
    return rm.dropna()

def rolling_average_onesided_hanning(data, window_length):
    hanning = signal.windows.hann(window_length * 2 -1)[0:window_length]
    hanning = hanning / sum(hanning)

    rolling_average = np.convolve(data,hanning,'same')

    return pd.Series(rolling_average, index=data.index)
