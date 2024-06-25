

from collections import OrderedDict, UserDict
from dataclasses import is_dataclass
from enum import Enum
import sys
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
        module_name = self.__class__.__module__
        for m_name, m_type in sys.modules.items():
            if module_name == m_name:
                class_globals = vars(m_type)
        type_hint_list = [typing.get_type_hints(cls, class_globals) for cls in self.__class__.mro()]
        type_hint_list.reverse()
        type_hints = {key: value for d in type_hint_list for key, value in d.items()}
        for attr, t in type_hints.items():
            object.__setattr__(self, attr, correct_type(self.__getattribute__(attr), t))
            

T = TypeVar('T')
def correct_type(input_value: Any, t: Type[T]) -> T:
    if type(input_value) == t:
        return input_value
    if hasattr(t, '_name'):
        if t._name == 'Union' and type(None) in t.__args__:
            if input_value:
                return correct_type(input_value, t.__args__[0])
            else:
                t = t.__args__[0]
                if hasattr(t, '_name'):
                    if t._name == 'Tuple':
                        return tuple()
                    elif t._name == 'List':
                        return list()
                    elif t._name == 'Dict':
                        return dict()
                    else:
                        raise NotImplementedError('Type Correction for Optional %s' % t._name)
                else:
                    return None
        if t._name == 'Tuple':
            return tuple([correct_type(x, t.__args__[0]) for x in input_value])
        elif t._name == 'List':
            return [correct_type(x, t.__args__[0]) for x in input_value]
        elif t._name == 'Dict':
            return dict({correct_type(key, t.__args__[0]): correct_type(value, t.__args__[1]) for key, value in input_value.items()})
        elif t._name == None:
            return None
        else:
            raise NotImplementedError('Type Correction for %s' % t._name)
    if is_dataclass(t):
        return t(**input_value)
    if issubclass(t, Enum):
        if issubclass(t, (str, int, float, bool)):
            return t(input_value)
        else:
            return t[input_value]
    return t(input_value)



