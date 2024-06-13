from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
from pathlib import Path
import sys
sys.path += [str(Path(__file__).parent.parent)]
from util.conversions import map_to_class

@dataclass(frozen=True)
class Degradation:
    '''General Degradation information'''
    system: float
    '''Total System DT50'''
    soil: float
    '''DT50 in soil'''
    surfaceWater: float
    '''DT50 in water'''
    sediment: float
    '''DT50 in sediment'''
    metabolites: Tuple['Degradation']
    '''Information on which metabolites will form'''

    def __post_init__(self):
        object.__setattr__(self, 'metabolites', tuple([map_to_class(x) for x in self.metabolites]))



@dataclass(frozen=True)
class Sorption:
    '''Information about the sorption behavior of a compound. Steps12 uses the koc, Pelmo uses all values'''
    koc: float
    freundlich: float

@dataclass(frozen=True)
class Compound:
    '''A Compound definition'''
    molarMass: float
    '''molar mass in g/mol'''
    waterSolubility: float
    sorption: Sorption
    '''A sorption behaviour'''
    degradation: Degradation
    '''Degradation behaviours'''
    parent: Optional['Compound'] = None
    '''The parent Compound, if any'''
    plant_uptake: float = 0
    '''Fraction of plant uptake'''
    

    def __post_init__(self):
        object.__setattr__(self, 'sorption', map_to_class(self.sorption, Sorption))
        object.__setattr__(self, 'degradation', map_to_class(self.degradation, Degradation))
        if self.parent is not None:
            object.__setattr__(self, 'parent', map_to_class(self.parent, Compound))



