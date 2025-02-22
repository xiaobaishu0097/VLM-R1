import importlib
import logging


class Register:
    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise RuntimeError(
                f"Value of a Registry must be a callable!\nValue: {value}"
            )
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning(f"Key {key} already in registry {self._name}.")
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()


class Registers:
    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    model = Register("model")
    agent = Register("agent")
    episode = Register("episode")
    optimizer = Register("optimizer")


# import optimizers

# ALL_MODULES = [
#     ("optimizers", optimizers.__all__),
# ]


def _handle_errors(errors):
    """Log out and possibly reraise errors during import."""
    if not errors:
        return
    for name, err in errors:
        logging.warning(f"Module {name} import failed: {err}")


def init_register():
    """Import all modules for register."""
    errors = []
    for base_dir, modules in ALL_MODULES:
        for name in modules:
            full_name = f"{base_dir}.{name}"
            try:
                importlib.import_module(base_dir, name)
                logging.debug(f"{full_name} loaded.")
            except ImportError as error:
                errors.append((name, error))
    _handle_errors(errors)
