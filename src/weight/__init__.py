from .ema import *

def get_weight_function(param):
    if param == "ema":
        return ema
    return None