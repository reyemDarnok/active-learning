from typing import TypeVar, Union, Dict, Type
from enum import Enum

T = TypeVar('T')

def map_to_class(obj: Union[T, Dict], cls: Type[T]) -> T:
    '''Turns the input obj into an instance of class cls.
    Assumes obj is either a map of constructor arguments or already an instance of cls'''
    if isinstance(obj, cls):
        return obj
    else:
        return cls(**obj)

E = TypeVar('E')

def str_to_enum(obj: Union[E, str], cls: Type[E]) -> E:
    '''Turn the input obj into an instance of the enum cls.
    Assumes the input is either an enum name, value or instance'''
    if isinstance(obj, cls):
        return obj
    else:
        try:
            return cls(obj)
        except ValueError:
            return cls[obj]