from .new_infections import *


def get_metric_function(param):
    if param == "new_infections":
        return new_infections
    return None
