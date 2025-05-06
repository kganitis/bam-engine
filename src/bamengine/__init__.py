from importlib.metadata import version as _v

__version__ = tuple(map(int, _v("bamengine").split(".")))
