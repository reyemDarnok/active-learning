from typing import Generator, TypeVar

T = TypeVar('T')


def repeat_infinite(value: T) -> Generator[T, None, None]:
    while True:
        yield value


def repeat_n_times(obj: T, times: int) -> Generator[T, None, None]:
    for _ in range(times):
        yield obj
