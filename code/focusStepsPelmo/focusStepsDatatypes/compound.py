from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict
from pathlib import Path
import sys
sys.path += [str(Path(__file__).parent.parent)]
from util.conversions import map_to_class, str_to_enum
from enum import Enum, auto


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
    metabolites: Dict[str, MetaboliteDegradation] = field(default_factory=dict)
    '''Information on which metabolites will form'''

    def __init__(self, system: float, soil: float, surfaceWater: float, sediment: float,
                 type: Union[DegradationType, int] = DegradationType.FACTORS,
                 metabolites: Dict[str, Union[MetaboliteDegradation, Dict[str, Union[float, int, Dict[str, float]]]]] = None,
                 photo: List[Union[Photodegradation, Dict[str, float]]] = None):
        self.system = system
        self.soil = soil
        self.surfaceWater = surfaceWater
        self.sediment = sediment
        self.type = str_to_enum(type, DegradationType)
        if metabolites is None:
            self.metabolites = []
        else:
            self.metabolites = {key: map_to_class(value, MetaboliteDegradation) for key, value in metabolites.items()} 
        if len(self.metabolites) < 5:
            self.metabolites += self.metabolites * (5 - len(self.metabolites))
        if photo is None:
            self.photo = []
        else:
            self.photo = [map_to_class(x, Photodegradation) for x in photo]



@dataclass
class Sorption:
    '''Information about the sorption behavior of a compound. Steps12 uses the koc, Pelmo uses all values'''
    koc: float
    freundlich: float

@dataclass
class Substance:
    '''A Compound definition'''
    molarMass: float
    '''molar mass in g/mol'''
    waterSolubility: float
    sorption: Sorption
    '''A list of soption behaviours'''
    degradation: Degradation
    '''Degradation behaviours'''
    parent: Optional['Substance'] = None
    '''The parent Compound, if any'''
    plant_uptake: float = 0
    '''Fraction of plant uptake'''
    

    def __init__(self, molarMass: float, waterSolubility: float, sorption: Union[Sorption, Dict[str, float]],
                 degradation: Union[Degradation, Dict[str, float]], 
                 parent: Union[Optional['Substance'], Dict[str, float]] = None,
                 plant_uptake: float = 0):
        self.molarMass = molarMass
        self.waterSolubility = waterSolubility
        self.sorption = map_to_class(sorption, Sorption)
        self.degradation = map_to_class(degradation, Degradation)
        if parent is None:
            self.parent = None
        else:
            self.parent = map_to_class(parent, Substance)
        self.plant_uptake = plant_uptake



