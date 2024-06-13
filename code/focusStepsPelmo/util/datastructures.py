

from collections import OrderedDict, UserDict
from typing import Any, Dict, Optional


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

class HashableRSDict(RSDict):
    def __hash__(self):
        return hash(frozenset(self.items()))

