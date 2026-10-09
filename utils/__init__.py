"""Shared calculations; simulator and plotting dependencies load on demand."""
from importlib import import_module
__all__ = ['debug', 'visualization']

def __getattr__(name):
    if name in __all__:
        module = import_module('.' + name, __name__)
        globals()[name] = module
        return module
    raise AttributeError(name)
