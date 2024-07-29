"""A file of useful method for converting objects between types"""
import re
from collections import OrderedDict, UserDict, UserList
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from enum import Enum
from json import JSONEncoder
from pathlib import Path
from typing import Any, Generator, Iterable, List, TypeVar, Union, Dict, Tuple

T = TypeVar('T')


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
        """Handles the cases that the Encoder does not recognize and returns something that it does.
        Note that this does not allow for a change in the default representation of objects that the default
        Encoder understands"""
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
            return {"seconds": o.total_seconds()}
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, frozenset):
            return list(o)
        if is_dataclass(o):
            return asdict(o)
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


def flatten(to_flatten: Union[List, Dict, Tuple, timedelta, Any]) -> Generator[str, None, None]:
    """Flatten a given structure to a csv row. 
    Calls str on anything that is not one of the following special cases
    
    dict
    
    list
    
    implements _asdict (NamedTuples, mostly)
    
    dataclass
    
    :param to_flatten: The structure to flatten"""
    if isinstance(to_flatten, dict):
        ordered = OrderedDict(sorted(to_flatten.items(), key=lambda x: x[0]))
        yield from (flattened for _, value in ordered.items() for flattened in flatten(value))
    elif isinstance(to_flatten, (list, tuple)):
        yield from (flattened for value in to_flatten for flattened in flatten(value))
    elif hasattr(to_flatten, 'asdict'):
        yield from flatten(to_flatten.asdict())
    elif is_dataclass(to_flatten):
        yield from flatten(asdict(to_flatten))
    elif type(to_flatten) == timedelta:
        yield to_flatten.total_seconds()
        yield to_flatten.microseconds
    else:
        to_flatten = str(to_flatten)
        yield re.sub(r'([\\,])', r"\\\1", to_flatten)


def excel_date_to_datetime(excel_date: int) -> datetime:
    """Converts an int in the internal representation of Excel to a datetime object
    :excel_date: The number of days since 1899-12-30 (Excels internal representation for dates)"""
    base_date = datetime(year=1899, month=12, day=30)
    return base_date + timedelta(excel_date)


def flatten_to_keys(o: Any, prefix=None) -> Generator[str, None, None]:
    """Convert an object into a List of table headings"""
    if prefix is None:
        prefix = []
    if isinstance(o, (dict, UserDict)):
        ordered = OrderedDict(sorted(o.items(), key=lambda x: x[0]))
        yield from flatten_dict_to_keys(ordered, prefix)
    elif isinstance(o, (list, UserList, tuple)):
        yield from flatten_list_to_keys(o, prefix)
    elif hasattr(o, 'asdict'):
        for key in flatten_to_keys(o.asdict(), prefix):
            yield key
    elif is_dataclass(o):
        for key in flatten_to_keys(asdict(o), prefix):
            yield key
    elif type(o) == timedelta:
        yield ".".join(prefix + ['seconds'])
        yield ".".join(prefix + ['microseconds'])
    else:
        yield ".".join(prefix)


def flatten_dict_to_keys(d: Dict, prefix=None) -> Generator[str, None, None]:
    """Convert a dict into table headings"""
    if prefix is None:
        prefix = []
    for key, value in d.items():
        for k in flatten_to_keys(value, prefix=prefix + [str(key)]):
            yield k


def flatten_list_to_keys(to_flatten: List, prefix=None) -> Generator[str, None, None]:
    """Convert a list into table headings"""
    if prefix is None:
        prefix = []
    for index, value in enumerate(to_flatten):
        for k in flatten_to_keys(value, prefix=prefix + [str(index)]):
            yield k


def uncomment(file: Path, comment_char: str = '#') -> Generator[str, None, None]:
    """Open a file and return it line by line, omitting any empty lines or lines whose first non-whitespace
    character is comment_char"""
    with file.open() as raw:
        for line in raw:
            if line.rstrip() and line.lstrip()[0] not in comment_char:
                yield line
