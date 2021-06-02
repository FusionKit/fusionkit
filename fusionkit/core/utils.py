# Utilities
# general numerical or pythonics utilities

import numpy as np

def number(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def find(val, arr,n=1):
    if isinstance(arr,list):
        arr_ = np.array(arr)
    else:
        arr_ = arr
    if n == 1:
        return np.argsort(np.abs(arr_-val))[0]
    else:
        return list(np.argsort(np.abs(arr_-val)))[:n]