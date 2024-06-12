

from collections import OrderedDict, UserDict
from typing import Any, Dict


class RSDict(UserDict):
    '''Dict ordered by reverse order of keys'''
    
    def __init__(self, source: Dict):
        key_order = reversed(sorted(source.keys()))
        data = OrderedDict()
        for key in key_order:
            data[key] = source[key]
        super().__init__(data)

