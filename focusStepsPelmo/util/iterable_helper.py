from typing import Generator, TypeVar

T = TypeVar('T')


def repeat_infinite(value: T) -> Generator[T, None, None]:
    """Generate the same object as many times as the Generator is queried
    :param value: The value to generate repeatedly
    :return: An infinite Generator repeating value
    >>> gen = repeat_infinite(5)
    >>> next(gen)
    5
    >>> next(gen)
    5
    >>> next(gen)
    5
    """
    while True:
        yield value


def repeat_n_times(value: T, times: int) -> Generator[T, None, None]:
    """Generate the same object a given number of times
    :param value: The value to generate repeatedly
    :param times: How often to generate value
    :return: A Generator generating the object value a times number of times
    >>> gen = repeat_n_times(5, 3)
    >>> list(gen)
    [5, 5, 5]
    """
    for _ in range(times):
        yield value


def count_up(start: int = 0) -> Generator[int, None, None]:
    """Generate ascending natural numbers
    :param start: The first number to generate
    :return: An infinite Generator of ascending natural numbers, starting at start
    >>> from_default = count_up()
    >>> next(from_default)
    0
    >>> next(from_default)
    1
    >>> next(from_default)
    2
    >>> from_100 = count_up(100)
    >>> next(from_100)
    100
    >>> next(from_100)
    101
    """
    while True:
        yield start
        start += 1
