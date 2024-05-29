from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Type, Union, NamedTuple


from .helperfunctions import map_to_class, str_to_enum

class Coverage(Enum):
    No = 0
    Minimal = auto()
    Average = auto()
    Full = auto()

class CropDataMixin(NamedTuple):
    display_name: str
    driftValues: List[float]
    interception: Dict[Coverage, float]

class Crop(CropDataMixin, Enum):
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

class Scenario(Enum):
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
    display_name: str
    defined_scenarios: List[Scenario]

class PelmoCrop(PelmoCropMixin, Enum):
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
    
class Region(Enum):
    NoRunoff = 0
    North = 1
    South = 2


@dataclass
class Application:
    rate: float
    number: int = 1
    interval: int = 1
    factor: float = 1

class Emergence(Enum):
    first_emergence = 0
    first_maturation = 1
    first_harvest = 2
    second_emregence = 3
    second_maturation = 4
    second_harvest = 5



class Season(Enum):
    Spring = auto()
    Summer = auto()
    Autumn = auto()

@dataclass
class Timing:
    emergence: Emergence
    offset: int
    season: Season

    def __init__(self, emergence: Union[Emergence, str, int], offset: int, season: Union[Season, str]):
        self.emergence = str_to_enum(emergence, Emergence)
        self.offset = offset
        self.season = str_to_enum(season, Season)

@dataclass
class GAP:
    modelCrop: Crop
    application: Application
    cropCover: Dict[Season, Coverage]
    regions: List[Region]
    timings: List[Timing] = field(default_factory=list)

    def __init__(self, modelCrop: Union[Crop, str],
                 application: Union[Application, Dict[str, float]],
                 cropCover: Union[Dict[Season, Coverage], Dict[str, str]],
                 regions: List[Union[Region, str]],
                 timings: List[Union[Timing, Dict[str, Union[int, str]]]] = None):
        if isinstance(modelCrop, Crop):
            self.modelCrop = modelCrop
        else:
            self.modelCrop = str_to_enum(modelCrop, Crop)
        self.application = map_to_class(application, Application)
        self.cropCover = {str_to_enum(season, Season): 
                          str_to_enum(coverage, Coverage) 
                          for season, coverage in cropCover.items()}
        self.regions = [str_to_enum(region, Region) for region in regions]
        if timings == None:
            self.timings = []
        else:
            self.timings = [map_to_class(x, Timing) for x in timings]

    @property
    def seasons(self):
        return self.cropCover.keys()
    
    @property
    def interception(self):
        return {season: self.modelCrop.interception[self.cropCover[season]] for season in self.seasons}
    
    @property
    def driftFactor(self):
        return self.modelCrop.driftValues[self.application.number - 1]

