from typing import Callable

cache_clears: list[Callable[[], None]] = []


def clear_caches():
    for cache_clear in cache_clears:
        cache_clear()
