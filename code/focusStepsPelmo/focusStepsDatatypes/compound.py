from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict
from pathlib import Path
import sys
sys.path += [str(Path(__file__).parent.parent)]
from util.conversions import map_to_class, str_to_enum
from enum import Enum, auto


@dataclass
class Occurrence:
    '''Used for Step12 calculations
    Describes the Occurrence behaviour of a substance.'''
    sediment: float
    '''The Occurrence in sediment. Is always 100 for parents and less for metabolites'''
    soil: float
    '''The Occurrence in soil. Is always 100 for parents and less for metabolites'''


class DegradationType(Enum):
    '''Used by Pelmo to describe the type of degdradation'''
    FACTORS = 0
    CONSTANT_WITH_DEPTH = auto()
    INDIVIDUAL = auto()
    FACTORS_LIQUID_PHASE = auto()
    CONSTANT_WITH_DEPTH_LIQUID_PHASE = auto()
    INDIVIDUAL_LIQUID_PHASE = auto()

@dataclass
class Moisture:
    '''Used by Pelmo'''
    absolute: float
    relative: float
    exp: float

@dataclass
class MetaboliteDegradation:
    '''Used by Pelmo to describe the degradation to metabolites'''
    rate: float
    '''The degradation rate. Calculated as ln(2)/DT50'''
    temperature: float
    q10: float
    moisture: Moisture
    stochiometric_factor: float
    rel_deg_new_sites: int

    def __init__(self, rate: float, temperature: float, q10: float, moisture: Moisture, stochiometric_factor: float, rel_deg_new_sites: int):
        self.rate = rate
        self.temperature = temperature
        self.q10 = q10
        self.moisture = map_to_class(moisture, Moisture)
        self.stochiometric_factor = stochiometric_factor
        self.rel_deg_new_sites = rel_deg_new_sites

@dataclass
class Photodegradation:
    '''Used by pelmo'''
    inverse_rate: float
    i_ref: float

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
    type: DegradationType = DegradationType.FACTORS
    '''The type of degradation for pelmo'''
    metabolites: Dict[str, MetaboliteDegradation] = field(default_factory=dict)
    '''Information on which metabolites will form'''
    photo: List[Photodegradation] = field(default_factory=list)
    '''Photodegradation information'''

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
class Volatization:
    '''Used by Pelmo'''
    henry: float
    solubility: float
    vaporization_pressure: float
    diff_air: float
    temperature: float

@dataclass
class Sorption:
    '''Information about the sorption behavior of a compound. Steps12 uses the koc, Pelmo uses all values'''
    koc: float
    freundlich: float
    pH: float 
    pKa: float
    limit_freundl: float
    annual_increment: float
    k_doc: float
    percent_change: float
    koc2: float
    pH2: float
    f_neq: float
    kdes: float

@dataclass
class Substance:
    '''A Compound definition'''
    molarMass: float
    '''molar mass in g/mol'''
    waterSolubility: float
    sorptions: List[Sorption]
    '''A list of soption behaviours'''
    degradation: Degradation
    '''Degradation behaviours'''
    maxOccurrence: Occurrence
    '''Simple formation fractions for Steps12'''
    parent: Optional['Substance'] = None
    '''The parent Compound, if any'''
    freundlich: float = 1
    plant_uptake: float = 0
    '''Fraction of plant uptake'''
    volatizations: List[Volatization] = field(default_factory = list)
    '''A list of volatization behaviours'''
    

    def __init__(self, molarMass: float, waterSolubility: float, sorption: Union[Sorption, Dict[str, float]],
                 degradation: Union[Degradation, Dict[str, float]], 
                 maxOccurrence: Union[Occurrence, Dict[str, float]],
                 parent: Union[Optional['Substance'], Dict[str, float]] = None,
                 freundlich: float = 1,
                 plant_uptake: float = 0,
                 volatizations: List[Union[Volatization, Dict[str, float]]] = None):
        self.molarMass = molarMass
        self.waterSolubility = waterSolubility
        self.sorption = map_to_class(sorption, Sorption)
        self.degradation = map_to_class(degradation, Degradation)
        self.maxOccurrence = map_to_class(maxOccurrence, Occurrence)
        if parent is None:
            self.parent = None
        else:
            self.parent = map_to_class(parent, Substance)
        self.freundlich = freundlich
        self.plant_uptake = plant_uptake
        if volatizations is None:
            self.volatizations = []
        else:
            self.volatizations = [map_to_class(x, Volatization) for x in volatizations]
        if len(self.volatizations) < 2:
            self.volatizations *= 2



