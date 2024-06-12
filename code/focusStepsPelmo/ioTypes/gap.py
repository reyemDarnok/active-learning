from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union, NamedTuple

from pathlib import Path
import sys
import typing
sys.path += [str(Path(__file__).parent.parent)]
from util.conversions import map_to_class, str_to_enum


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

class PrincipalStage(int, Enum):
    Germination = 0
    Leaf = 1
    Tillering = 2
    Elongation = 3
    Bolting = 4
    Inflorescence = 5
    Flowering = 6
    DevelopmentFruit =  7
    Maturity = 8
    Senescence = 9

class FOCUSCropMixin(NamedTuple):
    '''Crop information for Pelmo'''
    display_name: str
    '''The name to use for display'''
    defined_scenarios: List[Scenario]
    '''The scenarios that are defined for this Crop in Pelmo'''
    interception: typing.OrderedDict[PrincipalStage, float]
    '''Mapping bbch states to interception values'''

    def __init__(self, display_name: str, defined_scenarios: List[Scenario], interception: Dict[PrincipalStage, float]):
        self.display_name = display_name
        self.defined_scenarios = defined_scenarios
        key_order = reversed(sorted(interception.keys()))
        self.interception = OrderedDict()
        for key in key_order:
            self.interception[key] = interception[key]

s = PrincipalStage
class FOCUSCrop(FOCUSCropMixin, Enum):
    '''The crops defined for Pelmo. Each defined as a PelmoCropMixin'''
    AP = FOCUSCropMixin(display_name= "Apples", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S, Scenario.T],
                        interception={s.Maturity: 80, s.DevelopmentFruit: 70, s.Flowering: 65, s.Germination: 50})
    BB = FOCUSCropMixin(display_name= "Bush berries", 
                        defined_scenarios=[Scenario.J],
                        interception={s.Maturity: 80, s.DevelopmentFruit: 65, s.Flowering: 65, s.Germination: 50})
    BF = FOCUSCropMixin(display_name= "Beans (field)", 
                        defined_scenarios=[Scenario.H, Scenario.K, Scenario.N],
                        interception={s.Germination: 0, s.Leaf: 25, s.Tillering: 40, s.Flowering: 70, s.Senescence: 80})
    BV = FOCUSCropMixin(display_name= "Beans (vegetables)", 
                        defined_scenarios=[Scenario.O, Scenario.T],
                        interception={s.Germination: 0, s.Leaf: 25, s.Tillering: 40, s.Flowering: 70, s.Senescence: 80})
    CA = FOCUSCropMixin(display_name= "Carrots", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.O, Scenario.T],
                        interception={s.Germination: 0, s.Leaf: 25, s.Tillering: 60, s.Flowering: 80, s.Senescence: 80})
    CB = FOCUSCropMixin(display_name= "Cabbage", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.O, Scenario.S, Scenario.T],
                        interception={s.Germination: 0, s.Leaf: 25, s.Tillering: 40, s.Flowering: 70, s.Senescence: 90})
    CI = FOCUSCropMixin(display_name= "Citrus", 
                        defined_scenarios=[Scenario.P, Scenario.O, Scenario.S, Scenario.T],
                        interception={s.Germination: 70})
    CO = FOCUSCropMixin(display_name= "Cotton", 
                        defined_scenarios=[Scenario.S, Scenario.T],
                        interception={s.Germination: 0, s.Leaf: 30, s.Tillering: 60, s.Flowering: 75, s.Senescence: 90})
    GA = FOCUSCropMixin(display_name= "Grass and alfalfa", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S, Scenario.T],
                        interception={s.Germination: 0, s.Leaf: 40, s.Tillering: 60, s.Flowering: 70, s.Senescence: 90})
    LS = FOCUSCropMixin(display_name= "Linseed", 
                        defined_scenarios=[Scenario.N],
                        interception={s.Germination: 0, s.Leaf: 30, s.Tillering: 60, s.Flowering: 70, s.Senescence: 90})
    MZ = FOCUSCropMixin(display_name= "Maize", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S, Scenario.T],
                        interception={s.Germination: 0, s.Leaf: 25, s.Tillering: 50, s.Flowering: 75, s.Senescence: 90})
    ON = FOCUSCropMixin(display_name= "Onions", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.O, Scenario.T],
                        interception={s.Germination: 0, s.Leaf: 10, s.Tillering: 25, s.Flowering: 40, s.Senescence: 60})
    OS = FOCUSCropMixin(display_name= "Oilseed rape (summer)", 
                        defined_scenarios=[Scenario.J, Scenario.N, Scenario.O],
                        interception={s.Germination: 0, s.Leaf: 40, s.Tillering: 80, s.Flowering: 80, s.Senescence: 90})
    OW = FOCUSCropMixin(display_name= "Oilseed rape (winter)", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.K, Scenario.N, Scenario.P, Scenario.O],
                        interception={s.Germination: 0, s.Leaf: 40, s.Tillering: 80, s.Flowering: 80, s.Senescence: 90})
    PE = FOCUSCropMixin(display_name= "Peas (animals)", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.N],
                        interception={s.Germination: 0, s.Leaf: 35, s.Tillering: 55, s.Flowering: 85, s.Senescence: 85})
    PO = FOCUSCropMixin(display_name= "Potatoes", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S, Scenario.T],
                        interception={s.Germination: 0, s.Leaf: 15, s.Tillering: 50, s.Flowering: 80, s.Senescence: 50})
    SB = FOCUSCropMixin(display_name= "Sugar beets", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S, Scenario.T],
                        interception={s.Germination: 0, s.Leaf: 20, s.Tillering: 70, s.Flowering: 90, s.Senescence: 90})
    SC = FOCUSCropMixin(display_name= "Spring cereals", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.O],
                        interception={s.Germination: 0, s.Leaf: 25, s.Tillering: 50, s.Elongation: 70, s.Flowering: 90, s.Senescence: 90})
    SF = FOCUSCropMixin(display_name= "Sunflower", 
                        defined_scenarios=[Scenario.P, Scenario.S],
                        interception={s.Germination: 0, s.Leaf: 20, s.Tillering: 50, s.Flowering: 75, s.Senescence: 90})
    SO = FOCUSCropMixin(display_name= "Soybeans", 
                        defined_scenarios=[Scenario.P],
                        interception={s.Germination: 0, s.Leaf: 35, s.Tillering: 55, s.Flowering: 85, s.Senescence: 65})
    SW = FOCUSCropMixin(display_name= "Strawberries", 
                        defined_scenarios=[Scenario.H, Scenario.J, Scenario.K, Scenario.S],
                        interception={s.Germination: 0, s.Leaf: 30, s.Tillering: 50, s.Flowering: 60, s.Senescence: 60})
    TB = FOCUSCropMixin(display_name= "Tobacco", 
                        defined_scenarios=[Scenario.P, Scenario.T],
                        interception={s.Germination: 0, s.Leaf: 50, s.Tillering: 70, s.Flowering: 90, s.Senescence: 90})
    TM = FOCUSCropMixin(display_name= "Tomatoes", 
                        defined_scenarios=[Scenario.C, Scenario.P, Scenario.O, Scenario.S, Scenario.T],
                        interception={s.Germination: 0, s.Leaf: 50, s.Tillering: 70, s.Flowering: 80, s.Senescence: 50})
    VI = FOCUSCropMixin(display_name= "Vines", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.K, Scenario.P, Scenario.O, Scenario.S, Scenario.T],
                        interception={s.Maturity: 85, s.DevelopmentFruit: 70, s.Flowering: 60, s.Inflorescence: 50, s.Germination: 40},
                        interception={s.Germination: 0, s.Leaf: 25, s.Tillering: 40, s.Flowering: 70, s.Senescence: 80})
    WC = FOCUSCropMixin(display_name= "Winter cereals", 
                        defined_scenarios=[Scenario.C, Scenario.H, Scenario.J, Scenario.N, Scenario.P, Scenario.O],
                        interception={s.Germination: 0, s.Leaf: 25, s.Tillering: 50, s.Elongation: 70, s.Flowering: 90, s.Senescence: 90}) # K, S, T have crp files but are not officially defined there

    @staticmethod
    def from_acronym( acronym: str) -> 'FOCUSCrop':
        return FOCUSCrop[acronym]
    
    def get_interception(self, bbch: int) -> float:
        for key, value in self.interception.items():
            if bbch >= key:
                return value
        raise ValueError(f'No Interception found for bbch {bbch}')

@dataclass
class Timing:
    '''During which bbch stadium will the application happen'''
    bbch_state: int
    '''Relative to which development state'''

    def principal_stage(self):
        return PrincipalStage(min(9, self.bbch_state / 10))

@dataclass
class Application:
    '''Application information'''
    rate: float
    '''How much compound will be applied in g/ha'''
    number: int = 1
    '''How often will be applied'''
    interval: int = 1
    '''What is the minimum interval between applications'''
    factor: float = 1
    timing: Timing
    '''What is the timing of application'''

    def __init__(self, rate: float, number: int, interval: int, factor: float,
                 timing: Union[Timing, Dict[str, Union[int, str]]] = None):
        self.rate = rate
        self.number = number
        self.interval = interval
        self.factor = factor
        self.timing = map_to_class(timing, Timing)

@dataclass
class GAP:
    '''Defines a GAP'''
    modelCrop: FOCUSCrop
    '''The crop that the field is modelled after'''
    application: Application
    '''The values of the actual application'''
    

    def __init__(self, modelCrop: Union[FOCUSCrop, str],
                 application: Union[Application, Dict[str, float]],
                 ):
        if isinstance(modelCrop, FOCUSCrop):
            self.modelCrop = modelCrop
        else:
            self.modelCrop = str_to_enum(modelCrop, FOCUSCrop)
        self.application = map_to_class(application, Application)

    @property
    def seasons(self):
        """The seasons for which the gap is defined"""
        return list(self.cropCover.keys())
    
    @property
    def interception(self):
        """A dictionary mapping seasons to the crop interception in that season"""
        return {season: self.modelCrop.interception[self.cropCover[season]] for season in self.seasons}
    
    @property
    def driftFactor(self):
        """How much drift there is"""
        return self.modelCrop.driftValues[self.application.number - 1]

