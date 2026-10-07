import numpy as np

def cummean(a):
    cumsum = a.cumsum()
    cummean = cumsum/np.arange(1,len(a)+1)
    return(cummean)
