import dataclasses
from json import JSONEncoder
from typing import NamedTuple, TypeVar, Union, Dict, Type
from enum import Enum
import enum

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
        
class EnhancedJSONEncoder(JSONEncoder):
        def default(self, o):
            if hasattr(o, '_asdict'):
                return o._asdict()
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            if isinstance(o, Enum):
                if not isinstance(o, (str, int, float, bool)):
                    return o.name
            return super().default(o)
