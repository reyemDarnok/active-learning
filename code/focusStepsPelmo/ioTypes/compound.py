from dataclasses import dataclass, field
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
        object.__setattr__(self, 'system', float(self.system))
        object.__setattr__(self, 'soil', float(self.soil))
        object.__setattr__(self, 'surfaceWater', float(self.surfaceWater))
        object.__setattr__(self, 'sediment', float(self.sediment))


@dataclass(frozen=True)
class Sorption:
    '''Information about the sorption behavior of a compound. Steps12 uses the koc, Pelmo uses all values'''
    koc: float
    freundlich: float

    def __post_init__(self):
        object.__setattr__(self, 'koc', float(self.koc))
        object.__setattr__(self, 'freundlich', float(self.freundlich))


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
    metabolites: Dict[float, 'Compound'] = field(default_factory=dict)
    '''The parent Compound, if any'''
    plant_uptake: float = 0
    '''Fraction of plant uptake'''
    name: str = "Unknown Name"
    

    def __post_init__(self):
        object.__setattr__(self, 'molarMass', float(self.molarMass))
        object.__setattr__(self, 'waterSolubility', float(self.waterSolubility))
        object.__setattr__(self, 'plant_uptake', float(self.plant_uptake))
        object.__setattr__(self, 'sorption', map_to_class(self.sorption, Sorption))
        object.__setattr__(self, 'degradation', map_to_class(self.degradation, Degradation))
        if self.metabolites:
            object.__setattr__(self, 'metabolites', {float(formation_fraction): map_to_class(metabolite, Compound) for formation_fraction, metabolite in self.metabolites.items()})
        else:
            object.__setattr__(self, 'metabolites', {})



