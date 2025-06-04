"""A file for methods to generate objects from probabilistic definitions"""
import logging
import math
import random
import sys
from abc import ABC, abstractmethod
from collections import UserDict, UserList
from typing import Any, Generic, Tuple, Dict, List, Type, TypeVar

import numpy


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

T = TypeVar("T")
class Definition(ABC, Generic[T]):
    """An abstract parent class for a definition of a random object generation."""

    @property
    @abstractmethod
    def is_static(self) -> bool:
        # noinspection GrazieInspection
        """Whether two invocations of make_sample will differ
                :return: True if make_sample can be cached, False if it cannot"""
        pass

    @abstractmethod
    def make_sample(self) -> T:
        """Generate a single object from this definition
        :return: The generated object"""
        pass

    @abstractmethod
    def make_vector(self, obj: T) -> Tuple[float, ...]:
        """Create a tuple of floats describing the random choices taken in generating obj
        :param obj: An object generated from this definition
        :return: A tuple of floats in the range [-1,1]. While the length of the tuple differs between Definitions,
        for a single definition the length of the tuple will always be the same, regardless of the obj"""
        pass

    @staticmethod
    def parse(to_parse: Any) -> 'Definition[T]':
        """Parse a description of a definition into a tree of Definition objects. If some objects in the tree do not
        change between invocations of make_sample they will be transformed into LiteralDefinition objects to improve
        performance
        :param to_parse: The description to parse
        :return: The definition that resulted from to_parse
        TODO tests"""
        if isinstance(to_parse, (dict, UserDict)):
            if 'type' in to_parse.keys() and 'parameters' in to_parse.keys():
                refined: Dict[str, Any] = to_parse # type: ignore
                definition = TemplateDefinition[T].parse(refined)
            else:
                refined: Dict[Any, Any] = to_parse # type: ignore
                definition = DictDefinition(refined) # type: ignore
        elif isinstance(to_parse, (list, tuple, UserList)):
            definition: Definition[T] = ListDefinition(to_parse) # type: ignore
        else:
            definition: Definition[T] = LiteralDefinition(to_parse)
        if definition.is_static:
            return LiteralDefinition[T](definition.make_sample())
        else:
            return definition


class LiteralDefinition(Definition[T]):
    """A definition for a literal value
    >>> test = LiteralDefinition({"some": "value"})
    >>> test.is_static
    True
    >>> test.make_sample()
    {'some': 'value'}
    >>> test.make_vector(None)
    ()
    """

    def __init__(self, value: T):
        """
        :param value: The value this definition represents
        """
        self.value = value

    @property
    def is_static(self) -> bool:
        return True

    def make_sample(self) -> T:
        return self.value

    def make_vector(self, obj: T) -> Tuple[float, ...]:
        return tuple()

K = TypeVar("K")
V = TypeVar("V")
class DictDefinition(Definition[Dict[K, V]]):
    """A definition for a dictionary
    >>> test = DictDefinition({"key": {"type": "choices", "parameters": {"options": [1]}}})
    >>> test.is_static
    False
    >>> test.make_sample()
    {'key': 1}
    >>> test.make_vector({"key": 1})
    (0,)
    >>> static_test = DictDefinition({'k': 'v'})
    >>> static_test.is_static
    True"""

    def __init__(self, definition: Dict[K, V]):
        self.definition: Dict[K, Definition[V]] = {key: Definition.parse(value) for key, value in definition.items()}

    def make_sample(self) -> Dict[K, V]:
        return {key: value.make_sample() for key, value in self.definition.items()}

    def make_vector(self, obj: Dict[Any, Any]) -> Tuple[float, ...]:
        try:
            return tuple(val
                        for key, key_definition in self.definition.items()
                        for val in key_definition.make_vector(obj[key]))
        except:
            print(obj)
            raise

    @property
    def is_static(self) -> bool:
        for value in self.definition.values():
            if not value.is_static:
                return False
        return True


class ListDefinition(Definition[List[T]]):
    """ A definition for a List
    >>> test = ListDefinition([{"type": "choices", "parameters": {"options": [1]}},
    ... {"type": "choices", "parameters": {"options": [2]}}])
    >>> test.is_static
    False
    >>> test.make_sample()
    [1, 2]
    >>> test.make_vector([1,2])
    (0, 0)
    >>> static_test = ListDefinition([3,4,5])
    >>> static_test.is_static
    True"""

    def __init__(self, definition: List[Any]):
        self.definition: List[Definition[T]] = [Definition.parse(value) for value in definition]

    @property
    def is_static(self) -> bool:
        for value in self.definition:
            if not value.is_static:
                return False
        return True

    def make_sample(self) -> List[T]:
        return [value.make_sample() for value in self.definition]

    def make_vector(self, obj: List[T]) -> Tuple[float, ...]:
        return tuple(val
                     for item, item_definition in zip(obj, self.definition)
                     for val in item_definition.make_vector(item))


class TemplateDefinition(Definition[T], ABC):
    """The root definition for templates in the definitions"""

    @staticmethod
    def parse(to_parse: Dict[str, Any]) -> 'TemplateDefinition[T]':
        """Find the proper template requested by to_parse and instantiate it
        :param to_parse: The template definition
        :return: The definition for the template
        >>> t = TemplateDefinition.parse({"type": "choices", "parameters": {"options": []}})
        >>> isinstance(t, ChoicesDefinition)
        True
        >>> t = TemplateDefinition.parse({"type": "steps", "parameters": {"start": 0, "stop": 1}})
        >>> isinstance(t, StepsDefinition)
        True
        >>> t = TemplateDefinition.parse({"type": "random", "parameters": {"lower_bound": 0.3, "upper_bound": 0.7}})
        >>> isinstance(t, RandomDefinition)
        True
        >>> t = TemplateDefinition.parse({"type": "copies", "parameters": {"minimum": 0, "maximum": 2, "value": 4}})
        >>> isinstance(t, CopiesDefinition)
        True"""
        types: Dict[str, type[TemplateDefinition[T]]] = {
            'choices': ChoicesDefinition[T], # type: ignore pylance does not understand this type chooser
            'steps': StepsDefinition,
            'random': RandomDefinition,
            'copies': CopiesDefinition[T],
            'log_normal': LogNormalDefinition
        }
        return types[to_parse['type']](**to_parse['parameters'])

    @property
    def is_static(self) -> bool:
        return False


class LogNormalDefinition(TemplateDefinition[float]):
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def make_sample(self) -> float:
        return numpy.random.lognormal(mean=self.mu * math.log(10), sigma=self.sigma * math.log(10))

    def make_vector(self, obj: float) -> Tuple[float, ...]:
        if obj != 0:
            return (min(-1.0, max(1.0, (math.log10(obj) - self.mu) / self.sigma / 3)),)
        else:
            logging.getLogger().warn("LogNormalDefinition produced 0")
            return (0,)


class ChoicesDefinition(TemplateDefinition[T]):
    """ A definition for the choices template. It takes a list of options as a parameter and randomly chooses one
    when generating a sample. If the chosen option itself is again a template it will also be evaluated
    >>> import random
    >>> random.seed(42)
    >>> test = ChoicesDefinition([1,2,3])
    >>> test.is_static
    False
    >>> test.make_sample()
    3
    >>> test.make_vector(1)
    (-1.0,)
    >>> test = ChoicesDefinition([{"type": "random", "parameters": {"lower_bound": 1, "upper_bound": 2}}])
    >>> test.make_sample()
    1.025010755222667"""

    def __init__(self, options: List[Any]):
        self.options: List[Definition[T]] = [Definition.parse(option) for option in options]

    def make_sample(self) -> T:
        return random.choice(self.options).make_sample()

    def make_vector(self, obj: T) -> Tuple[float]:
        for index, option in enumerate(self.options):
            if option.is_static and obj == option.make_sample():
                return (normalize_feature(index, 0, len(self.options) - 1),)
        return (0,)


class StepsDefinition(TemplateDefinition[float]):
    # noinspection GrazieInspection
    """A definition for the steps template. It takes a range definition and a scale factor as arguments and generates
        a random value from these when making a sample."""

    def __init__(self, start: int, stop: int, step: int = 1, scale_factor: float = 1):
        """
        :param start: The step argument to range
        :param stop: The stop argument to range
        :param step: The step argument to range
        :param scale_factor: As floats are not supported by the underlying random,
        this is a multiplicative factor to the final result.
        I.e. if you want quarter steps from 0 to 1, call StepsDefinition(0,4,1,0.25)
        >>> import random
        >>> random.seed(42)
        >>> test = StepsDefinition(0,8,2,0.4)
        >>> test.is_static
        False
        >>> test.make_sample()
        0.0
        >>> test.make_vector(1.2)
        (-0.2500000000000001,)
        """
        self.start = start
        self.stop = stop
        self.step = step
        self.scale_factor = scale_factor

    def make_sample(self) -> float:
        return random.randrange(self.start, self.stop, self.step) * self.scale_factor

    def make_vector(self, obj: float) -> Tuple[float, ...]:
        return (normalize_feature(obj / self.scale_factor, self.start, self.stop),)


class RandomDefinition(TemplateDefinition[float]):
    """ A Definition for a random value. When making a sample, the random float can either be creating uniformly along
    the unit scaling or the log unit scaling
    >>> import random
    >>> random.seed(42)
    >>> test = RandomDefinition(0,10)
    >>> test.is_static
    False
    >>> test.make_sample()
    6.394267984578837
    >>> test.make_vector(3)
    (-0.4,)
    >>> log_test = RandomDefinition(1,1_000_000,True)
    >>> log_test.make_sample()
    1.4127474476060933
    >>> log_test.make_vector(100)
    (-0.33333333333333326,)"""

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


class CopiesDefinition(TemplateDefinition[List[T]]):
    """A Definition for copies of a value. If the value is itself a template, it will be evaluated for each copy
    generated when making a sample
    >>> import random
    >>> random.seed(42)
    >>> test = CopiesDefinition(minimum=2, maximum=4, value="a")
    >>> test.is_static
    False
    >>> test.make_sample()
    ['a', 'a', 'a', 'a']
    >>> test.make_vector(["s", "s"])
    (0.01168934382734399, -1.0)
    >>> template_val = CopiesDefinition(minimum=1, maximum=5,
    ... value={"type": "random", "parameters": {"lower_bound": 1, "upper_bound": 2}})
    >>> template_val.make_sample()
    [1.025010755222667]
    >>> template_val.make_vector([1.3, 1.4, 1.1])
    (-0.9846015770866611, 0.0)"""

    def __init__(self, minimum: int, maximum: int, value: Any):
        self.minimum = minimum
        self.maximum = maximum
        self.value: Definition[T] = Definition.parse(value)

    def make_sample(self) -> List[T]:
        return [self.value.make_sample() for _ in range(random.randint(self.minimum, self.maximum))]

    def make_vector(self, obj: List[T]) -> Tuple[float, float]:
        number = normalize_feature(len(obj), self.minimum, self.maximum)
        values = normalize_hash_feature(tuple(self.value.make_vector(copy) for copy in obj))
        return values, number
