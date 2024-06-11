from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Union, NamedTuple

from pathlib import Path
import sys
sys.path += [str(Path(__file__).parent.parent)]
from util.conversions import map_to_class, str_to_enum


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

class FOCUSCropMixin(NamedTuple):
    '''Crop Information for Steps12'''
    display_name: str
    '''The name of the crop for display purposes'''
    driftValues: List[float]
    '''How much substance drifts into the water'''


class FOCUSCrop(FOCUSCropMixin, Enum):
    '''The crops defined for Steps12, each defined as a CropDataMixin'''
    AA = FOCUSCropMixin(
        display_name = "Aerial appln",
        driftValues= [33.2, 33.2, 33.2, 33.2, 33.2, 33.2, 33.2, 33.2])
    CS = FOCUSCropMixin(
        display_name = "Cereals, spring",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    CW = FOCUSCropMixin(
        display_name = "Cereals, winter",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    CI = FOCUSCropMixin(
        display_name = "Citrus",
        driftValues= [15.725, 12.129, 11.011, 10.124, 9.743, 9.204, 9.102, 8.656])
    CO = FOCUSCropMixin(
        display_name = "Cotton",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    FB = FOCUSCropMixin(
        display_name = "Field beans",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    GA = FOCUSCropMixin(
        display_name = "Grass / alfalfa",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    HH = FOCUSCropMixin(
        display_name = "Hand high",
        driftValues= [8.028, 7.119, 6.898, 6.631, 6.636, 6.431, 6.227, 6.173])
    HL = FOCUSCropMixin(
        display_name = "Hand low",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    HP = FOCUSCropMixin(
        display_name = "Hops",
        driftValues= [19.326, 17.723, 15.928, 15.378, 15.114, 14.902, 14.628, 13.52])
    LG = FOCUSCropMixin(
        display_name = "Legumes",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    MZ = FOCUSCropMixin(
        display_name = "Maize",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    ND = FOCUSCropMixin(
        display_name = "No drift",
        driftValues= [0, 0, 0, 0, 0, 0, 0, 0])
    OS = FOCUSCropMixin(
        display_name = "OSR, spring",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    OW = FOCUSCropMixin(
        display_name = "OSR, winter",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    OL = FOCUSCropMixin(
        display_name = "Olives",
        driftValues= [15.725, 12.129, 11.011, 10.124, 9.743, 9.204, 9.102, 8.656])
    PE = FOCUSCropMixin(
        display_name = "P/S fruit, early",
        driftValues= [29.197, 25.531, 23.96, 23.603, 23.116, 22.76, 22.69, 22.241])
    PL = FOCUSCropMixin(
        display_name = "P/S fruit, late",
        driftValues= [15.725, 12.129, 11.011, 10.124, 9.743, 9.204, 9.102, 8.656])
    PS = FOCUSCropMixin(
        display_name = "Potatoes",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    SY = FOCUSCropMixin(
        display_name = "Soybeans",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    SB = FOCUSCropMixin(
        display_name = "Sugar beets",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    SU = FOCUSCropMixin(
        display_name = "Sunflowers",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    TB = FOCUSCropMixin(
        display_name = "Tobacco",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    VB = FOCUSCropMixin(
        display_name = "Veg. bulb",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    VF = FOCUSCropMixin(
        display_name = "Veg. fruiting",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    VL = FOCUSCropMixin(
        display_name = "Veg. leafy",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    VR = FOCUSCropMixin(
        display_name = "Veg. root",
        driftValues= [2.759, 2.438, 2.024, 1.862, 1.794, 1.631, 1.578, 1.512])
    VE = FOCUSCropMixin(
        display_name = "Vines, early",
        driftValues= [2.699, 2.496, 2.546, 2.499, 2.398, 2.336, 2.283, 2.265])
    VI = FOCUSCropMixin(
        display_name = "Vines, late",
        driftValues= [8.028, 7.119, 6.898, 6.631, 6.636, 6.431, 6.227, 6.173])
    
    @staticmethod
    def from_acronym( acronym: str) -> 'FOCUSCrop':
        return FOCUSCrop[acronym]

@dataclass
class Timing:
    '''During which bbch stadium will the application happen'''
    bbch_state: int
    '''Relative to which development state'''

    def date(self, crop: FOCUSCrop):
        raise RuntimeError('Not implemented')

@dataclass
class GAP:
    '''Defines a GAP'''
    modelCrop: FOCUSCrop
    '''The crop that the field is modelled after'''
    application: Application
    '''The values of the actual application'''
    timing: Timing
    '''What is the timing of application'''

    def __init__(self, modelCrop: Union[FOCUSCrop, str],
                 application: Union[Application, Dict[str, float]],
                 timing: Union[Timing, Dict[str, Union[int, str]]] = None):
        if isinstance(modelCrop, FOCUSCrop):
            self.modelCrop = modelCrop
        else:
            self.modelCrop = str_to_enum(modelCrop, FOCUSCrop)
        self.application = map_to_class(application, Application)
        self.timing = map_to_class(timing, Timing)

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

