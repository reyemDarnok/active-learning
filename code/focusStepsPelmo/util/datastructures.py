from collections import OrderedDict, UserDict
from dataclasses import is_dataclass
from enum import Enum
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import typing


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


def correct_type(input_value: Any, t: Type[T]) -> T:
    if type(input_value) == t:
        return input_value
    if hasattr(t, '__origin__'):  # Typing with Type Vars
        origin = t.__origin__
        # noinspection PyProtectedMember
        if origin == Union and type(None) in t.__args__:
            if input_value:
                return correct_type(input_value, t.__args__[0])
            else:
                t = t.__args__[0]
                return correct_type(None, t)
        elif origin == Tuple:
            if not input_value:
                return tuple()
            type_args = t.__args__
            if not isinstance(type_args[0], TypeVar):
                return tuple([correct_type(x, type_args[0]) for x in input_value])
            else:
                return tuple(input_value)
        elif origin == tuple:
            if input_value:
                return tuple(input_value)
            else:
                return tuple()
        elif origin == List:
            if not input_value:
                return list(input_value)
            type_args = t.__args__
            if not isinstance(type_args[0], TypeVar):
                return [correct_type(x, t.__args__[0]) for x in input_value]
            else:
                return list(input_value)
        elif origin == list:
            if input_value:
                return list(input_value)
            else:
                return list()
        elif origin == Dict:
            if not input_value:
                return {}
            type_args = t.__args__
            keys = input_value.keys()
            if not isinstance(type_args[0], TypeVar):
                keys = (correct_type(key, type_args[0]) for key in keys)
            if not isinstance(type_args[1], TypeVar):
                return {key: correct_type(input_value[key], type_args[1]) for key in keys}
            else:
                return {key: input_value[key] for key in keys}
        elif origin == dict:
            if origin:
                return dict(input_value)
            else:
                return dict()
        elif t._name is None:
            return None
        else:
            # noinspection PyProtectedMember
            raise NotImplementedError('Type Correction for %s' % t._name)
    if is_dataclass(t):
        return t(**input_value)
    if issubclass(t, Enum):
        if issubclass(t, (str, int, float, bool)):
            return t(input_value)
        else:
            return t[input_value]
    if input_value is None:
        return None
    return t(input_value)
