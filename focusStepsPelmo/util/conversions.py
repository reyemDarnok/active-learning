import dataclasses
import re
from collections import OrderedDict, UserDict
from datetime import datetime, timedelta
from enum import Enum
from json import JSONEncoder
from typing import Any, Generator, Iterable, List, TypeVar, Union, Dict, Type

T = TypeVar('T')


def map_to_class(obj: Union[T, Dict], cls: Type[T]) -> T:
    """Turns the input obj into an instance of class cls.
    Assumes obj is either a map of constructor arguments or already an instance of cls"""
    if isinstance(obj, cls):
        return obj
    else:
        return cls(**obj)


E = TypeVar('E')


def str_to_enum(obj: Union[E, str], cls: Type[E]) -> E:
    """Turn the input obj into an instance of the enum cls.
    Assumes the input is either an enum name, value or instance"""
    if isinstance(obj, cls):
        return obj
    else:
        try:
            return cls(obj)
        except ValueError:
            return cls[obj]


class EnhancedJSONEncoder(JSONEncoder):
    """Adds several previously not serializable classes to the JSON encoder
    Cannot be trivially extended to change the representation of serializable objects
    Currently adds support for:

    Objects implementing _asdict (primarily NamedTuples)

    dataclasses

    Enums

    UserDict
    """

    def default(self, o):
        if Enum in type(o).__bases__:
            print(f"Enum: {o}")
            if {str, int, float, bool} & set(type(o).__bases__):
                return o.value
            else:
                return o.name
        if hasattr(o, 'asdict'):
            return o.asdict()
        if hasattr(o, '_asdict'):
            # noinspection PyProtectedMember
            return o._asdict()
        if isinstance(o, timedelta):
            return {"microseconds": o.microseconds}
        if isinstance(o, frozenset):
            return list(o)
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, UserDict):
            return o.data
        return super().default(o)


def flatten_to_csv(to_flatten: Iterable[Dict[str, Any]]) -> Generator[str, None, None]:
    """Takes an Iterable of dictionaries and flattens them to csv data. 
    The fields are ordered alphabetically by their keys for dictionaries and by their index for lists.
    Nested structures are supported.
    :param to_flatten: The Iterable to flatten. Will be lazily iterated for each value of the generator"""
    for row in to_flatten:
        yield flatten(f"{row}\n")


def flatten(to_flatten: Union[List, Dict, Any]) -> str:
    """Flatten a given structure to a csv row. 
    Calls str on anything that is not one of the following special cases
    
    dict
    
    list
    
    implements _asdict (NamedTuples, mostly)
    
    dataclass
    
    :param to_flatten: The structure to flatten"""
    if isinstance(to_flatten, dict):
        ordered = OrderedDict(sorted(to_flatten.items()))
        return ",".join(flatten(value) for _, value in ordered.items())
    elif isinstance(to_flatten, list):
        return ",".join(flatten(value) for value in to_flatten)
    elif hasattr(to_flatten, '_asdict'):
        # noinspection PyProtectedMember
        return flatten(to_flatten._asdict())
    elif dataclasses.is_dataclass(to_flatten):
        return flatten(dataclasses.asdict(to_flatten))
    else:
        to_flatten = str(to_flatten)
        return re.sub(r'([\\,])', r"\\\1", to_flatten)


def excel_date_to_datetime(excel_date: int) -> datetime:
    base_date = datetime(year=1899, month=12, day=30)
    return base_date + timedelta(excel_date)
