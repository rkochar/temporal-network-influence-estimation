from .polyfit import *

def get_predict_function(param):
    if param == "polyfit":
        return polyfit
