import math
import random
import sys
from abc import ABC, abstractmethod
from collections import UserDict, UserList
from typing import Any, Tuple, Dict, List


def normalize_hash_feature(hashable: Any) -> float:
    """
    Normalize a hashable feature to a value in [-1, 1]
    :param hashable: The value to hash. Note that str hashes are not stable
    :return: The normalized value
    >>> import sys
    >>> normalize_hash_feature(tuple([1,2,3,4,5]))
    0.9015438605851986
    """
    return normalize_feature(hash(hashable), - 2 ** (sys.hash_info.width - 1), 2 ** (sys.hash_info.width - 1))


def normalize_feature(value: float, lower_bound: float, upper_bound: float) -> float:
    """Given the value of a feature and its possible range, normalize it into the [-1,1] range
    :param value: The feature value
    :param lower_bound: The lowest the value could have been
    :param upper_bound: The highest the value could have benn
    :return: A float in [-1, 1] that represents the value in its range
    >>> normalize_feature(1, 0, 4)
    -0.5
    >>> normalize_feature(100, -50, 200)
    0.19999999999999996
    """
    if upper_bound == lower_bound:
        return 0
    return ((value - lower_bound) / (upper_bound - lower_bound)) * 2 - 1


class Definition(ABC):
    @property
    @abstractmethod
    def is_static(self) -> bool:
        pass

    @abstractmethod
    def make_sample(self) -> Any:
        pass

    @abstractmethod
    def make_vector(self, obj: Any) -> Tuple[float, ...]:
        pass

    @staticmethod
    def parse(to_parse: Any) -> 'Definition':
        if isinstance(to_parse, (dict, UserDict)):
            if 'type' in to_parse.keys() and 'parameters' in to_parse.keys():
                definition = TemplateDefinition.parse(to_parse)
            else:
                definition = DictDefinition(to_parse)
        elif isinstance(to_parse, (list, tuple, UserList)):
            definition = ListDefinition(to_parse)
        else:
            definition = LiteralDefinition(to_parse)
        if definition.is_static:
            return LiteralDefinition(definition.make_sample())
        else:
            return definition


class LiteralDefinition(Definition):

    def __init__(self, value: Any):
        self.value = value

    @property
    def is_static(self) -> bool:
        return True

    def make_sample(self) -> Any:
        return self.value

    def make_vector(self, obj: Any) -> Tuple[float, ...]:
        return tuple()


class DictDefinition(Definition):
    def __init__(self, definition: Dict):
        self.definition = {key: Definition.parse(value) for key, value in definition.items()}

    def make_sample(self) -> Dict:
        return {key: value.make_sample() for key, value in self.definition.items()}

    def make_vector(self, obj: Dict) -> Tuple[float, ...]:
        return tuple(val
                     for key, key_definition in self.definition.keys()
                     for val in key_definition.make_vector(obj[key]))

    @property
    def is_static(self) -> bool:
        for value in self.definition.values():
            if not value.is_static:
                return False
        return True


class ListDefinition(Definition):

    def __init__(self, definition: List):
        self.definition = [Definition.parse(value) for value in definition]

    @property
    def is_static(self) -> bool:
        for value in self.definition:
            if not value.is_static:
                return False
        return True

    def make_sample(self) -> List:
        return [value.make_sample() for value in self.definition]

    def make_vector(self, obj: List) -> Tuple[float, ...]:
        pass


class TemplateDefinition(Definition, ABC):
    @staticmethod
    def parse(to_parse: Dict[str, Any]) -> 'Definition':
        types = {
            'choices': ChoicesDefinition,
            'steps': StepsDefinition,
            'random': RandomDefinition,
            'copies': CopiesDefinition
        }
        return types[to_parse['type']](**to_parse['parameters'])

    @property
    def is_static(self) -> bool:
        return False


class ChoicesDefinition(TemplateDefinition):
    def __init__(self, options: List):
        self.options = [Definition.parse(option) for option in options]

    def make_sample(self) -> Any:
        return random.choice(self.options).make_sample()

    def make_vector(self, obj: Any) -> Tuple[float]:
        for index, option in enumerate(self.options):
            if option.is_static and obj == option.make_sample():
                return (normalize_feature(index, 0, len(self.options) - 1),)
        return (0,)


class StepsDefinition(TemplateDefinition):
    def __init__(self, start: int, stop: int, step: int = 1, scale_factor: float = 1):
        self.start = start
        self.stop = stop
        self.step = step
        self.scale_factor = scale_factor

    def make_sample(self) -> float:
        return random.randrange(self.start, self.stop, self.step) * self.scale_factor

    def make_vector(self, obj: float) -> Tuple[float, ...]:
        return (normalize_feature(obj / self.scale_factor, self.start, self.stop),)


class RandomDefinition(TemplateDefinition[float]):
    def make_sample(self) -> float:
        if self.log_random:
            lower_bound = math.log(self.lower_bound)
            upper_bound = math.log(self.upper_bound)
            return math.exp(random.uniform(lower_bound, upper_bound))
        else:
            return random.uniform(self.lower_bound, self.upper_bound)

    def make_vector(self, obj: float) -> Tuple[float]:
        if self.log_random:
            lower_bound = math.log(self.lower_bound)
            upper_bound = math.log(self.upper_bound)
            obj = math.log(obj)
            return (normalize_feature(obj, lower_bound, upper_bound),)
        else:
            return (normalize_feature(obj, self.lower_bound, self.upper_bound),)

    def __init__(self, lower_bound: float, upper_bound: float, log_random: bool = False):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.log_random = log_random


class CopiesDefinition(TemplateDefinition):
    def __init__(self, minimum: int, maximum: int, value: Any):
        self.minimum = minimum
        self.maximum = maximum
        self.value = Definition.parse(value)

    def make_sample(self) -> List:
        return [self.value.make_sample() for _ in range(random.randint(self.minimum, self.maximum))]

    def make_vector(self, obj: List) -> Tuple[float, float]:
        number = normalize_feature(len(obj), self.minimum, self.maximum)
        values = normalize_hash_feature(tuple(self.value.make_vector(copy) for copy in obj))
        return values, number
