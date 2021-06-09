def filter_kwargs(func, kwargs):
    """Returns kwargs subset where only keys recognized by func are left."""
    return {key: value for key, value in kwargs.items() if key in func.__code__.co_varnames}


