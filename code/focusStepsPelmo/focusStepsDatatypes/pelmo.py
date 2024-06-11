from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import re
from typing import List, NamedTuple

from util.conversions import map_to_class


@dataclass
class WaterHorizon():
    '''Class describing a horizon segment line of a WASSER.PLM'''
    horizon: int
    compartment: int
    previous_storage: float
    previous_storage_soil_water_content: float
    leaching_input: float
    transpiration: float
    leaching_output: float
    current_storage: float
    current_storage_soil_water_content: float
    temperature: float
    def __init__(self, line: str):
        '''Expects a single line of the WASSER.PLM from a horizon segment'''
        segments = line.split()
        self.horizon = int(segments[0])
        self.compartment = int(segments[1])
        if len(segments) < 10 and '(' in segments[2]:
            self.previous_storage = float(segments[2].split('(')[0])
            self.previous_storage_soil_water_content = float(segments[2].split('(')[1].replace(')', ''))
            segments = segments[3:]
        else:
            self.previous_storage = float(segments[2])
            self.previous_storage_soil_water_content = float(segments[3].replace('(', '').replace(')', ''))
            segments = segments[4:]
        self.leaching_input = float(segments[0])
        self.transpiration = float(segments[1])
        self.leaching_output = float(segments[2])
        if len(segments) < 6:
            self.current_storage = float(segments[3].split('(')[0])
            self.current_storage_soil_water_content = float(segments[3].split('(')[1].replace(')', ''))
            segments = segments[4:]
        else:
            self.current_storage = float(segments[3])
            self.current_storage_soil_water_content = float(segments[4].replace('(', '').replace(')', ''))
            segments = segments[5:]
        self.temperature = segments[0]

@dataclass
class WaterPLM():
    '''A parsed form of a WASSER.PLM file in the Pelmo output. Does not yet parse all fields'''
    horizons: List[List[WaterHorizon]]
    '''A 2d List of horizons. Accessed as horizons[year][compartment]'''
    def __init__(self, file: Path) -> None:
        '''Expects the path to the WASSER.PLM'''
        years = [[[line for line in section.splitlines()[:-1] if line] for section in re.split(r"---+", year)[1:]] for year in file.read_text().split("ANNUAL WATER OUTPUT")[1:]]
        self.horizons = [[WaterHorizon(line) for line in year[2][:-1]] for year in years]

@dataclass
class ChemHorizon():
    '''Class describing a horizon segment line of a CHEM.PLM'''
    horizon : int
    compartment : int
    soil_application : float
    previous_storage : float
    leaching_input : float
    decay : float
    gas_diffusion : float
    plant_uptake : float
    leaching_output : float
    current_storage : float
    storage_in_neq_domain : float


    def __init__(self, line: str):
        '''Expects a single line of the CHEM.PLM from a horizon segment'''
        segments = line.split()
        self.horizon = int(segments[0])
        self.compartment = int(segments[1])
        self.soil_application = float(segments[2]) # might be bool, is written as 1.000/0.0000 value
        self.previous_storage = float(segments[3])
        self.leaching_input = float(segments[4])
        self.decay = float(segments[5])
        self.gas_diffusion = float(segments[6])
        self.plant_uptake = float(segments[7])
        self.leaching_output = float(segments[8])
        self.current_storage = float(segments[9])
        self.storage_in_neq_domain = float(segments[10])

@dataclass
class ChemPLM:
    '''A parsed form of a CHEM.PLM file in the Pelmo output. Does not yet parse all fields'''
    horizons: List[List[ChemHorizon]]
    '''A 2d List of horizons. Accessed as horizons[year][compartment]'''

    def __init__(self, file: Path) -> None:
        '''Expects the path to a CHEM.PLM file'''
        years = [[[line for line in section.splitlines()[:-1] if line] for section in re.split(r"---+", year)[1:]] for year in file.read_text().split("ANNUAL")[1:]]
        self.horizons = [[ChemHorizon(line) for line in year[2][:-1] if not '*' in line] for year in years]

class ApplicationType(int, Enum):
    """The different types of application Pelmo recognizes"""
    soil = 1
    linear = 2
    exp_foliar = 3
    manual = 4

@dataclass
class PelmoResult:
    psm: str
    scenario: str
    crop: str
    pec: List[float]


class Scenario(Enum):
    '''The Pelmo Scenarios. The key is the one letter shorthand and the value the full name'''
    C = "Châteaudun"
    H = "Hamburg"
    J = "Jokioinen"
    K = "Kremsmünster"
    N = "Okehampton"
    P = "Piacenza"
    O = "Porto"
    S = "Sevilla"
    T = "Thiva"

class PelmoCropMixin(NamedTuple):
    '''Crop information for Pelmo'''
    display_name: str
    '''The name to use for display'''
    defined_scenarios: List[Scenario]

    '''The scenarios that are defined for this Crop in Pelmo'''
    def toJSON(self):
        return {"display_name": self.display_name,
                "defined_scenarios": self.defined_scenarios}

class PelmoCrop(PelmoCropMixin, Enum):
    '''The crops defined for Pelmo. Each defined as a PelmoCropMixin'''
    AP = PelmoCropMixin(display_name= "Apples", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S, Scenario.T])
    BB = PelmoCropMixin(display_name= "Bush berries", 
                        defined_scenarios=[Scenario.J])
    BF = PelmoCropMixin(display_name= "Beans (field)", 
                        defined_scenarios=[Scenario.H, Scenario.K, Scenario.N])
    BV = PelmoCropMixin(display_name= "Beans (vegetables)", 
                        defined_scenarios=[Scenario.O, Scenario.T])
    CA = PelmoCropMixin(display_name= "Carrots", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.O, Scenario.T])
    CB = PelmoCropMixin(display_name= "Cabbage", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.O, Scenario.S, Scenario.T])
    CI = PelmoCropMixin(display_name= "Citrus", 
                        defined_scenarios=[Scenario.P, Scenario.O, Scenario.S, Scenario.T])
    CO = PelmoCropMixin(display_name= "Cotton", 
                        defined_scenarios=[Scenario.S, Scenario.T])
    GA = PelmoCropMixin(display_name= "Grass and alfalfa", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S, Scenario.T])
    LS = PelmoCropMixin(display_name= "Linseed", 
                        defined_scenarios=[Scenario.N])
    MZ = PelmoCropMixin(display_name= "Maize", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S, Scenario.T])
    ON = PelmoCropMixin(display_name= "Onions", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.O, Scenario.T])
    OS = PelmoCropMixin(display_name= "Oilseed rape (summer)", 
                        defined_scenarios=[Scenario.J, Scenario.N, Scenario.O])
    OW = PelmoCropMixin(display_name= "Oilseed rape (winter)", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.K, Scenario.N, Scenario.P, Scenario.O])
    PE = PelmoCropMixin(display_name= "Peas (animals)", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.N])
    PO = PelmoCropMixin(display_name= "Potatoes", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S, Scenario.T])
    SB = PelmoCropMixin(display_name= "Sugar beets", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S, Scenario.T])
    SC = PelmoCropMixin(display_name= "Spring cereals", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.O])
    SF = PelmoCropMixin(display_name= "Sunflower", 
                        defined_scenarios=[Scenario.P, Scenario.S])
    SO = PelmoCropMixin(display_name= "Soybeans", 
                        defined_scenarios=[Scenario.P])
    SW = PelmoCropMixin(display_name= "Strawberries", 
                        defined_scenarios=[Scenario.H, Scenario.J, Scenario.K, Scenario.S])
    TB = PelmoCropMixin(display_name= "Tobacco", 
                        defined_scenarios=[Scenario.P, Scenario.T])
    TM = PelmoCropMixin(display_name= "Tomatoes", 
                        defined_scenarios=[Scenario.C, Scenario.P, Scenario.O, Scenario.S, Scenario.T])
    VI = PelmoCropMixin(display_name= "Vines", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.K, Scenario.P, Scenario.O, Scenario.S, Scenario.T])
    WC = PelmoCropMixin(display_name= "Winter cereals", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.N, Scenario.P, Scenario.O]) # K, S, T have crp files but are not officially defined there

    @staticmethod
    def from_acronym( acronym: str) -> 'PelmoCrop':
        return PelmoCrop[acronym]
    
    def toJSON(self):
        return {"display_name": self.display_name,
                "defined_scenarios": self.defined_scenarios}
    

class Emergence(int, Enum):
    '''The possible application crop development timings for Pelmo'''
    first_emergence = 0
    first_maturation = 1
    first_harvest = 2
    second_emregence = 3
    second_maturation = 4
    second_harvest = 5

# Compound starts here
class DegradationType(int, Enum):
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