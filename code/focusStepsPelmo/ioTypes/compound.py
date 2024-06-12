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

    def __init__(self, system: float, soil: float, surfaceWater: float, sediment: float,
                 metabolites: List[Union['Degradation', Dict[str, Union[float, 'Degradation']]]] = None,
                 ):
        self.system = system
        self.soil = soil
        self.surfaceWater = surfaceWater
        self.sediment = sediment
        self.metabolites = [map_to_class(x) for x in metabolites]



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
    

    def __init__(self, molarMass: float, waterSolubility: float, sorption: Union[Sorption, Dict[str, float]],
                 degradation: Union[Degradation, Dict[str, float]], 
                 parent: Union[Optional['Compound'], Dict[str, float]] = None,
                 plant_uptake: float = 0):
        self.molarMass = molarMass
        self.waterSolubility = waterSolubility
        self.sorption = map_to_class(sorption, Sorption)
        self.degradation = map_to_class(degradation, Degradation)
        if parent is None:
            self.parent = None
        else:
            self.parent = map_to_class(parent, Compound)
        self.plant_uptake = plant_uptake



