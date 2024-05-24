from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Union, NamedTuple


from .helperfunctions import map_to_class, str_to_enum

class Coverage(Enum):
    No = 0
    Minimal = auto()
    Average = auto()
    Full = auto()

class CropDataMixin(NamedTuple):
    name: str
    driftValues: List[float]
    interception: Dict[Coverage, float]


class Crop(CropDataMixin, Enum):
    
    CS = CropDataMixin(
            name = "Cereals, spring",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.00, Coverage.Average: 0.20, Coverage.Full: 0.70})
    CW = CropDataMixin(
            name = "Cereals, winter",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.00, Coverage.Average: 0.20, Coverage.Full: 0.70})
    CI = CropDataMixin(
            name = "Citrus",
            driftValues= [15.725, 12.129, 11.011, 10.124, 9.743, 9.204, 9.102, 8.656],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.80, Coverage.Average: 0.80, Coverage.Full: 0.80})
    CO = CropDataMixin(
            name = "Cotton",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.30, Coverage.Average: 0.60, Coverage.Full: 0.75})
    FB = CropDataMixin(
            name = "Field beans",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.25, Coverage.Average: 0.40, Coverage.Full: 0.70})
    GA = CropDataMixin(
            name = "Grass / alfalfa",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.40, Coverage.Average: 0.60, Coverage.Full: 0.75})
    HP = CropDataMixin(
            name = "Hops",
            driftValues= [19.326, 17.723, 15.928, 15.378, 15.114, 14.902, 14.628, 13.52],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.50, Coverage.Full: 0.70})
    LG = CropDataMixin(
            name = "Legumes",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.25, Coverage.Average: 0.50, Coverage.Full: 0.70})
    MZ = CropDataMixin(
            name = "Maize",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.25, Coverage.Average: 0.50, Coverage.Full: 0.75})
    OS = CropDataMixin(
            name = "OSR, spring",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.40, Coverage.Average: 0.70, Coverage.Full: 0.75})
    OW = CropDataMixin(
            name = "OSR, winter",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.40, Coverage.Average: 0.70, Coverage.Full: 0.75})
    OL = CropDataMixin(
            name = "Olives",
            driftValues= [15.725, 12.129, 11.011, 10.124, 9.743, 9.204, 9.102, 8.656],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.70, Coverage.Average: 0.70, Coverage.Full: 0.70})
    PE = CropDataMixin(
            name = "P/S fruit, early",
            driftValues= [29.197, 25.531, 23.96, 23.603, 23.116, 22.76, 22.69, 22.241],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.40, Coverage.Full: 0.65})
    PL = CropDataMixin(
            name = "P/S fruit, late",
            driftValues= [15.725, 12.129, 11.011, 10.124, 9.743, 9.204, 9.102, 8.656],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.40, Coverage.Full: 0.65})
    PS = CropDataMixin(
            name = "Potatoes",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.15, Coverage.Average: 0.50, Coverage.Full: 0.70})
    SY = CropDataMixin(
            name = "Soybeans",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.50, Coverage.Full: 0.75})
    SB = CropDataMixin(
            name = "Sugar beets",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.70, Coverage.Full: 0.75})
    SU = CropDataMixin(
            name = "Sunflowers",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.50, Coverage.Full: 0.75})
    TB = CropDataMixin(
            name = "Tobacco",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.70, Coverage.Full: 0.75})
    VB = CropDataMixin(
            name = "Veg. bulb",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.10, Coverage.Average: 0.25, Coverage.Full: 0.40})
    VF = CropDataMixin(
            name = "Veg. fruiting",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.25, Coverage.Average: 0.50, Coverage.Full: 0.70})
    VL = CropDataMixin(
            name = "Veg. leafy",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.25, Coverage.Average: 0.40, Coverage.Full: 0.70})
    VR = CropDataMixin(
            name = "Veg. root",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.25, Coverage.Average: 0.50, Coverage.Full: 0.70})
    VE = CropDataMixin(
            name = "Vines, early",
            driftValues= [2.699, 2.496, 2.546, 2.499, 2.398, 2.336, 2.283, 2.265],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.40, Coverage.Average: 0.50, Coverage.Full: 0.60})
    VI = CropDataMixin(
            name = "Vines, late",
            driftValues= [8.028, 7.119, 6.898, 6.631, 6.636, 6.431, 6.227, 6.173],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.40, Coverage.Average: 0.50, Coverage.Full: 0.60})
    AA = CropDataMixin(
            name = "Aerial appln",
            driftValues= [33.2, 33.2, 33.2, 33.2, 33.2, 33.2, 33.2, 33.2],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.50, Coverage.Full: 0.70})
    HL = CropDataMixin(
            name = "Hand low",
            driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.50, Coverage.Full: 0.70})
    HH = CropDataMixin(
            name = "Hand high",
            driftValues= [8.028, 7.119, 6.898, 6.631, 6.636, 6.431, 6.227, 6.173],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.20, Coverage.Average: 0.50, Coverage.Full: 0.70})
    ND = CropDataMixin(
            name = "No drift",
            driftValues= [0, 0, 0, 0, 0, 0, 0, 0],
            interception = {Coverage.No: 0, Coverage.Minimal: 0.00, Coverage.Average: 0.00, Coverage.Full: 0.00})

    
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

class Season(Enum):
    Spring = auto()
    Summer = auto()
    Autumn = auto()

@dataclass
class GAP:
    modelCrop: Crop
    application: Application
    cropCover: Dict[Season, Coverage]
    regions: List[Region]

    def __init__(self, modelCrop: Union[Crop, str],
                 application: Union[Application, Dict[str, float]],
                 cropCover: Union[Dict[Season, Coverage], Dict[str, str]],
                 regions: Union[List[Region], List[str]]):
        if isinstance(modelCrop, Crop):
            self.modelCrop = modelCrop
        else:
            self.modelCrop = Crop[modelCrop]
        self.application = map_to_class(application, Application)
        self.cropCover = {str_to_enum(season, Season): 
                          str_to_enum(coverage, Coverage) 
                          for season, coverage in cropCover.items()}
        self.regions = [str_to_enum(region, Region) for region in regions]

    @property
    def seasons(self):
        return self.cropCover.keys()
    
    @property
    def interception(self):
        return {season: self.modelCrop.interception[self.cropCover[season]] for season in self.seasons}
    
    @property
    def driftFactor(self):
        return self.modelCrop.driftValues[self.application.number - 1]

