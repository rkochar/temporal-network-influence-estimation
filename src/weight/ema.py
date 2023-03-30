import numpy as np
from scipy.signal import lfilter


def ema(data, window=0):
    alpha = .1 # smoothing coefficient
    zi = [data[0]] # seed the filter state with first value
    # filter can process blocks of continuous data if <zi> is maintained
    y, zi = lfilter([1.-alpha], [1., -alpha], data, zi=zi)
    return y


# def ema(data, window=0):
#     alpha = 2 /(window + 1.0)
#     alpha_rev = 1-alpha
#     n = data.shape[0]
#
#     pows = alpha_rev**(np.arange(n+1))
#
#     scale_arr = 1/pows[:-1]
#     offset = data[0]*pows[1:]
#     pw0 = alpha*alpha_rev**(n-1)
#
#     mult = data*pw0*scale_arr
#     cumsums = mult.cumsum()
#     out = offset + cumsums*scale_arr[::-1]
#     return out