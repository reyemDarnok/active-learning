

from collections import OrderedDict, UserDict
from dataclasses import is_dataclass
from typing import Any, Dict, Optional, Type, TypeVar
import typing


class RSDict(UserDict):
    '''Dict ordered by reverse order of keys'''
    
    def __init__(self, source: Optional[Dict] = None):
        if source == None:
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
        type_hints = typing.get_type_hints(self)
        for attr, t in type_hints.items():
            object.__setattr__(self, attr, correct_type(self.__getattribute__(attr), t))
            

T = TypeVar('T')
def correct_type(input_value: Any, t: Type[T]) -> T:
    if type(input_value) == t:
        return input_value
    if hasattr(t, '_name'):
        if t._name == 'Tuple':
            return tuple([correct_type(x, t.__args__[0]) for x in input_value])
        if t._name == 'List':
            return [correct_type(x, t.__args__[0]) for x in input_value]
        if t._name == 'Dict':
            return dict({correct_type(key, t.__args__[0]): correct_type(value, t.__args__[1]) for key, value in input_value.items()})
    if is_dataclass(t):
        return t(**input_value)
    return t(input_value)



