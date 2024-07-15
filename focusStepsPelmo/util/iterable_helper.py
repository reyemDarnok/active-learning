from typing import Generator, TypeVar

T = TypeVar('T')


def repeat_infinite(value: T) -> Generator[T, None, None]:
    """Generate the same object as many times as the Generator is queried
    :param value: The value to generate repeatedly
    :return: An infinite Generator repeating value"""
    while True:
        yield value


def repeat_n_times(value: T, times: int) -> Generator[T, None, None]:
    """Generate the same object a given number of times
    :param value: The value to generate repeatedly
    :param times: How often to generate value
    :return: A Generator generating the object value a times number of times"""
    for _ in range(times):
        yield value


def count_up(start: int = 0) -> Generator[int, None, None]:
    """Generate ascending natural numbers
    :param start: The first number to generate
    :return: An infinite Generator of ascending natural numbers, starting at start"""
    while True:
        yield start
        start += 1
