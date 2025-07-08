import time
from typing import Callable, Any, Tuple


class FunctionTimer:
    def __init__(self, func: Callable):
        self.func = func
        self.__name__ = func.__name__

    def __call__(self, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
        start_time = time.perf_counter()
        result = self.func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        return result, duration
