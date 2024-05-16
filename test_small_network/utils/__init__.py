from . import my_csv
from . import dataset_manipulation
from . import model_manipulation
from . import timehistory
from . import train
from . import train_6b
from . import train_0
from . import train_1
from . import train_2

'''
import pkgutil
import importlib

# Iterate over all modules in the current package
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    # Import the module
    module = importlib.import_module('.' + module_name, package=__name__)
    
    # Add its functions to the package's namespace
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if callable(attribute):
            globals()[attribute_name] = attribute
'''