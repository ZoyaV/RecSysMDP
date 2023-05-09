import importlib.util
import sys


def lazy_import(name, package=None):
    module = sys.modules.get(name, None)
    if module is not None:
        return module

    spec = importlib.util.find_spec(name, package=package)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module
