import numpy as np
import json
import os
import importlib

def save_params(path, name, params):
    json.dump(params, open(os.path.join(path, name + '.txt'), 'w'))


def import_func(lib, func):
    return getattr(importlib.import_module(lib), func)