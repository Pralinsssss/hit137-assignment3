import time
import functools

def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMER] {func.__name__} took {end-start:.2f}s")
        return result
    return wrapper

def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        model_name = getattr(args[0], 'model_name', 'unknown') if args else 'unknown'
        print(f"[LOG] Calling {func.__name__} on {model_name}")
        return func(*args, **kwargs)
    return wrapper
