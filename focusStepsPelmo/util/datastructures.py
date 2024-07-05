import itertools
import sys
import typing
from collections import OrderedDict, UserDict
from dataclasses import is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union


class RSDict(UserDict):
    """Dict ordered by reverse order of keys"""

    def __init__(self, source: Optional[Dict] = None):
        if source is None:
            source = {}
        key_order = reversed(sorted(source.keys()))
        data = OrderedDict()
        for key in key_order:
            data[key] = source[key]
        super().__init__(data)


class HashableDict(UserDict):
    def __hash__(self):
        return hash(frozenset(self.items()))


class HashableRSDict(RSDict):
    def __hash__(self):
        return hash(frozenset(self.items()))


class TypeCorrecting:
    def __post_init__(self):
        module_name = self.__class__.__module__
        class_globals = {}
        for m_name, m_type in sys.modules.items():
            if module_name == m_name:
                class_globals = vars(m_type)
                break
        type_hint_list = [typing.get_type_hints(cls, class_globals) for cls in self.__class__.mro()]
        type_hint_list.reverse()
        type_hints = {key: value for d in type_hint_list for key, value in d.items()}
        for attr, t in type_hints.items():
            object.__setattr__(self, attr, correct_type(self.__getattribute__(attr), t))


T = TypeVar('T')


class NoValidStrategyException(Exception):
    def __init__(self, *args, strategy_exceptions: List[Exception]):
        super().__init__(*args)
        self.strategy_exceptions = strategy_exceptions


def correct_type(input_value: Any, t: Type[T]) -> T:
    if t == typing.Any:
        return input_value
    if hasattr(t, '__origin__'):  # Typing with Type Vars
        origin = t.__origin__
        # noinspection PyProtectedMember
        if origin == Union:
            if type(None) in t.__args__ and input_value is None:
                return None
            else:
                exceptions = []
                for union_type in t.__args__:
                    try:
                        return correct_type(input_value, union_type)
                    except Exception as e:
                        exceptions.append(e)
                raise NoValidStrategyException("Could not find a type that works with the value",
                                               strategy_exceptions=exceptions)
        elif origin == tuple:
            if not input_value:
                return tuple()
            if hasattr(t, '__args__') and not isinstance(t.__args__[0], TypeVar):
                type_args = t.__args__
                result = tuple()
                last_type = type_args[-2] if len(type_args) > 1 else type_args[1]
                for value, value_type in itertools.zip_longest(input_value, type_args, fillvalue=last_type):
                    if value_type != Ellipsis:
                        result += (correct_type(value, value_type),)
                    else:
                        if value == last_type:
                            break  # Ellipsis is not taken, i.e. a tuple of one or more ints has one int
                        else:
                            result += (correct_type(value, last_type),)
                return result
            else:
                return tuple(input_value)
        elif origin == list:
            if not input_value:
                return list()
            if hasattr(t, '__args__') and not isinstance(t.__args__[0], TypeVar):
                return [correct_type(x, t.__args__[0]) for x in input_value]
            else:
                return list(input_value)
        elif origin == dict:
            if not input_value:
                return {}
            keys = input_value.keys()
            if hasattr(t, '__args__'):
                type_args = t.__args__
                if not isinstance(type_args[0], TypeVar):
                    keys = (correct_type(key, type_args[0]) for key in keys)
                if not isinstance(type_args[1], TypeVar):
                    return {key: correct_type(input_value[key], type_args[1]) for key in keys}
            return {key: input_value[key] for key in keys}
        elif t._name is None:
            return None
        else:
            # noinspection PyProtectedMember
            raise NotImplementedError('Type Correction for %s' % t._name)
    if isinstance(input_value, t):
        return input_value
    if input_value is None:
        return None
    if hasattr(t, 'parse'):
        return t.parse(input_value)
    if is_dataclass(t):
        return t(**input_value)
    if issubclass(t, Enum):
        try:
            return t[input_value]
        except KeyError:
            return t(input_value)
    return t(input_value)
