from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Union, NamedTuple

from pathlib import Path
import sys
sys.path += [str(Path(__file__).parent.parent)]
from util.conversions import map_to_class, str_to_enum

class Coverage(int, Enum):
    '''Coverage categories for Steps12'''
    No = 0
    Minimal = auto()
    Average = auto()
    Full = auto()

class CropDataMixin(NamedTuple):
    '''Crop Information for Steps12'''
    display_name: str
    '''The name of the crop for display purposes'''
    driftValues: List[float]
    '''How much substance drifts into the water'''
    interception: Dict[Coverage, float]
    '''At which coverage level, how much substance does the plant intercept?'''

    def toJSON(self):
        return {"display_name": self.display_name,
                "driftValues": self.driftValues,
                "interception": self.interception}

class Crop(CropDataMixin, Enum):
    '''The crops defined for Steps12, each defined as a CropDataMixin'''
    AA = CropDataMixin(
        display_name = "Aerial appln",
        driftValues= [33.2, 33.2, 33.2, 33.2, 33.2, 33.2, 33.2, 33.2],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.50, Coverage.Full: 0.70})
    CS = CropDataMixin(
        display_name = "Cereals, spring",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.00, Coverage.Average: 0.20, Coverage.Full: 0.70})
    CW = CropDataMixin(
        display_name = "Cereals, winter",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.00, Coverage.Average: 0.20, Coverage.Full: 0.70})
    CI = CropDataMixin(
        display_name = "Citrus",
        driftValues= [15.725, 12.129, 11.011, 10.124, 9.743, 9.204, 9.102, 8.656],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.80, Coverage.Average: 0.80, Coverage.Full: 0.80})
    CO = CropDataMixin(
        display_name = "Cotton",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.30, Coverage.Average: 0.60, Coverage.Full: 0.75})
    FB = CropDataMixin(
        display_name = "Field beans",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.25, Coverage.Average: 0.40, Coverage.Full: 0.70})
    GA = CropDataMixin(
        display_name = "Grass / alfalfa",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.40, Coverage.Average: 0.60, Coverage.Full: 0.75})
    HH = CropDataMixin(
        display_name = "Hand high",
        driftValues= [8.028, 7.119, 6.898, 6.631, 6.636, 6.431, 6.227, 6.173],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.50, Coverage.Full: 0.70})
    HL = CropDataMixin(
        display_name = "Hand low",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.50, Coverage.Full: 0.70})
    HP = CropDataMixin(
        display_name = "Hops",
        driftValues= [19.326, 17.723, 15.928, 15.378, 15.114, 14.902, 14.628, 13.52],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.50, Coverage.Full: 0.70})
    LG = CropDataMixin(
        display_name = "Legumes",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.25, Coverage.Average: 0.50, Coverage.Full: 0.70})
    MZ = CropDataMixin(
        display_name = "Maize",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.25, Coverage.Average: 0.50, Coverage.Full: 0.75})
    ND = CropDataMixin(
        display_name = "No drift",
        driftValues= [0, 0, 0, 0, 0, 0, 0, 0],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.00, Coverage.Average: 0.00, Coverage.Full: 0.00})
    OS = CropDataMixin(
        display_name = "OSR, spring",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.40, Coverage.Average: 0.70, Coverage.Full: 0.75})
    OW = CropDataMixin(
        display_name = "OSR, winter",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.40, Coverage.Average: 0.70, Coverage.Full: 0.75})
    OL = CropDataMixin(
        display_name = "Olives",
        driftValues= [15.725, 12.129, 11.011, 10.124, 9.743, 9.204, 9.102, 8.656],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.70, Coverage.Average: 0.70, Coverage.Full: 0.70})
    PE = CropDataMixin(
        display_name = "P/S fruit, early",
        driftValues= [29.197, 25.531, 23.96, 23.603, 23.116, 22.76, 22.69, 22.241],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.40, Coverage.Full: 0.65})
    PL = CropDataMixin(
        display_name = "P/S fruit, late",
        driftValues= [15.725, 12.129, 11.011, 10.124, 9.743, 9.204, 9.102, 8.656],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.40, Coverage.Full: 0.65})
    PS = CropDataMixin(
        display_name = "Potatoes",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.15, Coverage.Average: 0.50, Coverage.Full: 0.70})
    SY = CropDataMixin(
        display_name = "Soybeans",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.50, Coverage.Full: 0.75})
    SB = CropDataMixin(
        display_name = "Sugar beets",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.70, Coverage.Full: 0.75})
    SU = CropDataMixin(
        display_name = "Sunflowers",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.50, Coverage.Full: 0.75})
    TB = CropDataMixin(
        display_name = "Tobacco",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.70, Coverage.Full: 0.75})
    VB = CropDataMixin(
        display_name = "Veg. bulb",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.10, Coverage.Average: 0.25, Coverage.Full: 0.40})
    VF = CropDataMixin(
        display_name = "Veg. fruiting",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.25, Coverage.Average: 0.50, Coverage.Full: 0.70})
    VL = CropDataMixin(
        display_name = "Veg. leafy",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.25, Coverage.Average: 0.40, Coverage.Full: 0.70})
    VR = CropDataMixin(
        display_name = "Veg. root",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.25, Coverage.Average: 0.50, Coverage.Full: 0.70})
    VE = CropDataMixin(
        display_name = "Vines, early",
        driftValues= [2.699, 2.496, 2.546, 2.499, 2.398, 2.336, 2.283, 2.265],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.40, Coverage.Average: 0.50, Coverage.Full: 0.60})
    VI = CropDataMixin(
        display_name = "Vines, late",
        driftValues= [8.028, 7.119, 6.898, 6.631, 6.636, 6.431, 6.227, 6.173],
        interception = {Coverage.No: 0, Coverage.Minimal: 0.40, Coverage.Average: 0.50, Coverage.Full: 0.60})
    
    @staticmethod
    def from_acronym( acronym: str) -> 'Crop':
        return Crop[acronym]
    
    def toJSON(self):
        return {"display_name": self.display_name,
                "driftValues": self.driftValues,
                "interception": self.interception}

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
    def from_acronym( acronym: str) -> 'Crop':
        return PelmoCrop[acronym]
    
    def toJSON(self):
        return {"display_name": self.display_name,
                "defined_scenarios": self.defined_scenarios}
    
class Region(int, Enum):
    '''The Steps12 Regions'''
    NoRunoff = 0
    North = 1
    South = 2


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

class Emergence(int, Enum):
    '''The possible application crop development timings for Pelmo'''
    first_emergence = 0
    first_maturation = 1
    first_harvest = 2
    second_emregence = 3
    second_maturation = 4
    second_harvest = 5



class Season(int, Enum):
    '''The Step12 Seasons'''
    Spring = auto()
    Summer = auto()
    Autumn = auto()

@dataclass
class Timing:
    '''A timing definition in Pelmo'''
    emergence: Emergence
    '''Relative to which development state'''
    offset: int
    '''How many days after (or before) that state'''
    season: Season
    '''In which Season is that'''

    def __init__(self, emergence: Union[Emergence, str, int], offset: int, season: Union[Season, str]):
        self.emergence = str_to_enum(emergence, Emergence)
        self.offset = offset
        self.season = str_to_enum(season, Season)

@dataclass
class GAP:
    '''Defines a GAP'''
    modelCrop: Crop
    '''The crop that the field is modelled after'''
    application: Application
    '''The values of the actual application'''
    cropCover: Dict[Season, Coverage]
    '''How much crop cover is in different Seasons'''
    regions: List[Region]
    '''Which regions are defined'''
    timing:Timing
    '''What is the timing of application'''

    def __init__(self, modelCrop: Union[Crop, str],
                 application: Union[Application, Dict[str, float]],
                 cropCover: Union[Dict[Season, Coverage], Dict[str, str]],
                 regions: List[Union[Region, str]],
                 timing: Union[Timing, Dict[str, Union[int, str]]] = None):
        if isinstance(modelCrop, Crop):
            self.modelCrop = modelCrop
        else:
            self.modelCrop = str_to_enum(modelCrop, Crop)
        self.application = map_to_class(application, Application)
        self.cropCover = {str_to_enum(season, Season): 
                          str_to_enum(coverage, Coverage) 
                          for season, coverage in cropCover.items()}
        self.regions = [str_to_enum(region, Region) for region in regions]
        self.timing = map_to_class(timing, Timing)

    def _asdict(self):
        return {"modelCrop": self.modelCrop, "application": self.application, "cropCover": self.cropCover, "regions": self.regions, "timing": self.timing}

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

