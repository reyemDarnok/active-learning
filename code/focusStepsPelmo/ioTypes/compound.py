from dataclasses import dataclass
from typing import List, Optional, Union, Dict
from pathlib import Path
import sys
sys.path += [str(Path(__file__).parent.parent)]
from util.conversions import map_to_class

@dataclass
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
    metabolites: List['Degradation']
    '''Information on which metabolites will form'''

    def __post_init__(self):
        self.metabolites = [map_to_class(x) for x in self.metabolites]



@dataclass
class Sorption:
    '''Information about the sorption behavior of a compound. Steps12 uses the koc, Pelmo uses all values'''
    koc: float
    freundlich: float

@dataclass
class Compound:
    '''A Compound definition'''
    molarMass: float
    '''molar mass in g/mol'''
    waterSolubility: float
    sorption: Sorption
    '''A list of soption behaviours'''
    degradation: Degradation
    '''Degradation behaviours'''
    parent: Optional['Compound'] = None
    '''The parent Compound, if any'''
    plant_uptake: float = 0
    '''Fraction of plant uptake'''
    

    def __post_init__(self):
        self.sorption = map_to_class(self.sorption, Sorption)
        self.degradation = map_to_class(self.degradation, Degradation)
        if self.parent is not None:
            self.parent = map_to_class(self.parent, Compound)



