def is_debug_mode():
    try:
        import pydevd

        return True
    except ImportError:
        return False
