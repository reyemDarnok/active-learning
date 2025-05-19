"""A file for useful data structures"""
import itertools
import sys
import typing
from collections import OrderedDict, UserDict
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Protocol, Type, TypeVar, Union

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
_T_contra = TypeVar("_T_contra", contravariant=True)
class SupportsDunderLT(Protocol[_T_contra]):
    def __lt__(self, other: _T_contra, /) -> bool: ...

class SupportsDunderGT(Protocol[_T_contra]):
    def __gt__(self, other: _T_contra, /) -> bool: ...
K_sortable = TypeVar('K_sortable', bound=Union[SupportsDunderLT[Any], SupportsDunderGT[Any]]) # type: ignore - from source in python

class RSDict(OrderedDict[K_sortable, V]):
    """Dict ordered by reverse order of keys"""

    def __init__(self, source: Optional[Dict[K_sortable, V]] = None):
        if source is None:
            source = dict[K_sortable, V]()
        key_order = reversed(sorted(source.keys()))
        data: OrderedDict[K_sortable, V] = OrderedDict()
        for key in key_order:
            data[key] = source[key]
        super().__init__(data)


class HashableDict(UserDict[K, V]):
    """A Dict that supports hashing. Note that using it as keys and then changing it will have unpredictable effects"""

    def __hash__(self):
        return hash(frozenset(self.items()))


class HashableRSDict(RSDict[K_sortable, V]):
    """A hashable version of RSDict. Note that using it as keys and then changing it will have unpredictable effects"""

    def __hash__(self):
        return hash(frozenset(self.items()))


class TypeCorrecting:
    """A class that enforces the type annotations of its members.
    Its intended use is as a superclass for dataclasses"""

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




class NoValidStrategyException(Exception):
    """There were several strategies for an operation specified but none worked"""

    def __init__(self, *args: Any, strategy_exceptions: List[Exception]):
        super().__init__(*args)
        self.strategy_exceptions = strategy_exceptions


def correct_type(input_value: Any, t: Type[T]) -> T:
    """
    Coerce the input_value into a value of type t. Respects all typing annotations with simple constructors
    """
    try:
        if t == typing.Any: # type: ignore - t comes from type annotations -> may well be any
            return input_value
        if hasattr(t, '__origin__'):  # Typing with Type Vars
            origin: Type[Any] = t.__origin__ # type: ignore - we know that __origin__ is always type information
            # noinspection PyProtectedMember
            if origin == Union:
                if type(None) in t.__args__: # type: ignore - every union has information which types are in the union
                    if input_value is None:
                        return None # type: ignore - if we reach here, None is a valic value of T
                exceptions: List[Exception] = []
                union_types: List[Type[Any]] = t.__args__ # type: ignore 
                for union_type in union_types:
                    try:
                        return correct_type(input_value, union_type)
                    except Exception as e:
                        exceptions.append(e)
                raise NoValidStrategyException("Could not find a type that works with the value",
                                               strategy_exceptions=exceptions)
            elif origin == tuple:
                if not input_value:
                    raise TypeError(f"Missing values for tuple construction")
                try:
                    if hasattr(t, '__args__') and not isinstance(t.__args__[0], TypeVar): # type: ignore 
                        type_args: List[Type[Any]] = t.__args__ # type: ignore 
                        result: List[Any] = list()
                        last_type: Type[Any] = type_args[-2] if len(type_args) > 1 else type_args[1] # type: ignore 
                        for value, value_type in itertools.zip_longest(input_value, type_args, fillvalue=last_type): # type: ignore 
                            if value_type != Ellipsis:
                                result.append(correct_type(value, value_type)) # type: ignore 
                            else:
                                if value == last_type:
                                    break  # Ellipsis is not taken, i.e. a tuple of one or more ints has one int
                                else:
                                    result.append(correct_type(value, last_type)) # type: ignore 
                        return tuple(result) # type: ignore 
                    else:
                        return tuple(input_value) # type: ignore 
                except TypeError:  # input_value is not iterable
                    if hasattr(t, '__args__') and not isinstance(t.__args__[0], TypeVar): # type: ignore
                        type_args: List[Type[Any]] = t.__args__ # type: ignore
                        if len(type_args) == 1 or (len(type_args) == 2 and type_args[1] == Ellipsis):
                            return (correct_type(input_value, type_args[0]),) # type: ignore
                    else:
                        return (input_value,) # type: ignore
            elif origin == set:
                if not input_value:
                    return set() # type: ignore
                try:
                    if hasattr(t, '__args__') and not isinstance(t.__args__[0], TypeVar): # type: ignore
                        type_args = t.__args__ # type: ignore
                        return {correct_type(x, type_args[0]) for x in input_value} # type: ignore
                    else:    
                        return set(input_value) # type: ignore
                except TypeError:
                    if hasattr(t, '__args__') and not isinstance(t.__args__[0], TypeVar): # type: ignore
                        type_args = t.__args__ # type: ignore
                        return {correct_type(input_value, type_args[0])} # type: ignore
                    else:
                        return {input_value} # type: ignore
            elif origin == frozenset:
                if not input_value:
                    return frozenset() # type: ignore
                try:
                    if hasattr(t, '__args__') and not isinstance(t.__args__[0], TypeVar): # type: ignore
                        type_args = t.__args__ # type: ignore
                        return frozenset(correct_type(x, type_args[0]) for x in input_value) # type: ignore
                    else:
                        return frozenset(input_value) # type: ignore
                except TypeError:
                    if hasattr(t, '__args__') and not isinstance(t.__args__[0], TypeVar): # type: ignore
                        type_args = t.__args__ # type: ignore
                        return frozenset({correct_type(input_value, type_args[0])}) # type: ignore
                    else:
                        return frozenset({input_value}) # type: ignore
            elif origin == list:
                if not input_value:
                    return list() # type: ignore
                try:
                    if hasattr(t, '__args__') and not isinstance(t.__args__[0], TypeVar): # type: ignore
                        return [correct_type(x, t.__args__[0]) for x in input_value] # type: ignore
                    else:
                        return list(input_value) # type: ignore
                except TypeError:  # input_value is not iterable
                    if hasattr(t, '__args__') and not isinstance(t.__args__[0], TypeVar): # type: ignore
                        return [correct_type(input_value, t.__args__[0])] # type: ignore
                    else:
                        return [input_value] # type: ignore
            elif origin == dict: 
                if not input_value:
                    return {} # type: ignore
                access_keys = input_value.keys()
                correct_keys = input_value.keys()
                if hasattr(t, '__args__'):
                    type_args = t.__args__ # type: ignore
                    if not isinstance(type_args[0], TypeVar):
                        correct_keys = (correct_type(key, type_args[0]) for key in access_keys)
                    if not isinstance(type_args[1], TypeVar):
                        return {correct_key: correct_type(input_value[access_key], type_args[1]) # type: ignore
                                for correct_key, access_key in zip(correct_keys, access_keys)}
                return {key: input_value[key] for key in access_keys} # type: ignore
            elif t._name is None: # type: ignore
                return None # type: ignore
            else:
                # noinspection PyProtectedMember
                raise NotImplementedError('Type Correction for %s' % t._name) # type: ignore
        # After the typing evaluation because isinstance errors if given a type hint as class instead of returning False
        if isinstance(input_value, t):
            return input_value
        if input_value is None:
            return None # type: ignore
        if hasattr(t, 'parse'):
            return t.parse(input_value) # type: ignore
        # noinspection PyBroadException
        try:
            return t(**input_value)
        except Exception:
            pass
        if issubclass(t, Enum):
            try:
                return t[input_value]
            except KeyError:
                return t(input_value)
        return t(input_value) # type: ignore
    except Exception as e:
        raise TypeError(f"Could not convert {input_value} to type {t}") from e
