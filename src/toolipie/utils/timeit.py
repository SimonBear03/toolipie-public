import time
from contextlib import contextmanager


@contextmanager
def timeit():
    start = time.perf_counter()
    try:
        yield lambda: time.perf_counter() - start
    finally:
        pass
