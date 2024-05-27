from typing import TypeVar, Union, Dict, Type
from enum import Enum

T = TypeVar('T')

def map_to_class(obj: Union[T, Dict], cls: Type[T]) -> T:
    if isinstance(obj, cls):
        return obj
    else:
        return cls(**obj)

E = TypeVar('E')

def str_to_enum(obj: Union[E, str], cls: Type[E]) -> E:
    if isinstance(obj, cls):
        return obj
    else:
        try:
            return cls(obj)
        except ValueError:
            return cls[obj]