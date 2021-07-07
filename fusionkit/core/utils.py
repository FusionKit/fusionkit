# Utilities
# general numerical or pythonics utilities

import numpy as np
from numpy.lib import gradient

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

def calcz(x,y):
    z = (1/y)*np.gradient(y,x,edge_order=2)
    return -z
