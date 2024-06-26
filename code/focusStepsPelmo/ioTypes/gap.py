from dataclasses import asdict, dataclass, replace
from enum import Enum
import math
from typing import Dict, List, OrderedDict, Tuple, Union, NamedTuple

from pathlib import Path
import sys
import typing

import pandas
sys.path += [str(Path(__file__).parent.parent)]
from util.datastructures import HashableRSDict, TypeCorrecting
from util.conversions import map_to_class, str_to_enum


class Scenario(str, Enum):
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
    Unplanted = -1
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
    defined_scenarios: Tuple[Scenario]
    '''The scenarios that are defined for this Crop in Pelmo'''
    interception: OrderedDict[PrincipalStage, float]
    '''Mapping bbch states to interception values'''


s = PrincipalStage
class FOCUSCrop(FOCUSCropMixin, Enum):
    '''The crops defined for Pelmo. Each defined as a PelmoCropMixin'''
    AP = FOCUSCropMixin(display_name= "Apples", 
                        defined_scenarios=tuple([Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S, Scenario.T]),
                        interception=HashableRSDict({s.Maturity: 80, s.DevelopmentFruit: 70, s.Flowering: 65, s.Germination: 50}))
    BB = FOCUSCropMixin(display_name= "Bush berries", 
                        defined_scenarios=tuple([Scenario.J]),
                        interception=HashableRSDict({s.Maturity: 80, s.DevelopmentFruit: 65, s.Flowering: 65, s.Germination: 50}))
    BF = FOCUSCropMixin(display_name= "Beans (field)", 
                        defined_scenarios=tuple([Scenario.H, Scenario.K, Scenario.N]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 25, s.Tillering: 40, s.Flowering: 70, s.Senescence: 80}))
    BV = FOCUSCropMixin(display_name= "Beans (vegetables)", 
                        defined_scenarios=tuple([Scenario.O, Scenario.T]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 25, s.Tillering: 40, s.Flowering: 70, s.Senescence: 80}))
    CA = FOCUSCropMixin(display_name= "Carrots", 
                        defined_scenarios=tuple([Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.O, Scenario.T]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 25, s.Tillering: 60, s.Flowering: 80, s.Senescence: 80}))
    CB = FOCUSCropMixin(display_name= "Cabbage", 
                        defined_scenarios=tuple([Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.O, Scenario.S, Scenario.T]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 25, s.Tillering: 40, s.Flowering: 70, s.Senescence: 90}))
    CI = FOCUSCropMixin(display_name= "Citrus", 
                        defined_scenarios=tuple([Scenario.P, Scenario.O, Scenario.S, Scenario.T]),
                        interception=HashableRSDict({s.Germination: 70}))
    CO = FOCUSCropMixin(display_name= "Cotton", 
                        defined_scenarios=tuple([Scenario.S, Scenario.T]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 30, s.Tillering: 60, s.Flowering: 75, s.Senescence: 90}))
    GA = FOCUSCropMixin(display_name= "Grass and alfalfa", 
                        defined_scenarios=tuple([Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S, Scenario.T]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 40, s.Tillering: 60, s.Flowering: 70, s.Senescence: 90}))
    LS = FOCUSCropMixin(display_name= "Linseed", 
                        defined_scenarios=tuple([Scenario.N]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 30, s.Tillering: 60, s.Flowering: 70, s.Senescence: 90}))
    MZ = FOCUSCropMixin(display_name= "Maize", 
                        defined_scenarios=tuple([Scenario.C, Scenario.H, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S, Scenario.T]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 25, s.Tillering: 50, s.Flowering: 75, s.Senescence: 90}))
    ON = FOCUSCropMixin(display_name= "Onions", 
                        defined_scenarios=tuple([Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.O, Scenario.T]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 10, s.Tillering: 25, s.Flowering: 40, s.Senescence: 60}))
    OS = FOCUSCropMixin(display_name= "Oilseed rape (summer)", 
                        defined_scenarios=tuple([Scenario.J, Scenario.N, Scenario.O]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 40, s.Tillering: 80, s.Flowering: 80, s.Senescence: 90}))
    OW = FOCUSCropMixin(display_name= "Oilseed rape (winter)", 
                        defined_scenarios=tuple([Scenario.C, Scenario.H, Scenario.K, Scenario.N, Scenario.P, Scenario.O]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 40, s.Tillering: 80, s.Flowering: 80, s.Senescence: 90}))
    PE = FOCUSCropMixin(display_name= "Peas (animals)", 
                        defined_scenarios=tuple([Scenario.C, Scenario.H, Scenario.J, Scenario.N]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 35, s.Tillering: 55, s.Flowering: 85, s.Senescence: 85}))
    PO = FOCUSCropMixin(display_name= "Potatoes", 
                        defined_scenarios=tuple([Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S, Scenario.T]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 15, s.Tillering: 50, s.Flowering: 80, s.Senescence: 50}))
    SB = FOCUSCropMixin(display_name= "Sugar beets", 
                        defined_scenarios=tuple([Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S, Scenario.T]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 20, s.Tillering: 70, s.Flowering: 90, s.Senescence: 90}))
    SC = FOCUSCropMixin(display_name= "Spring cereals", 
                        defined_scenarios=tuple([Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.O]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 25, s.Tillering: 50, s.Elongation: 70, s.Flowering: 90, s.Senescence: 90}))
    SF = FOCUSCropMixin(display_name= "Sunflower", 
                        defined_scenarios=tuple([Scenario.P, Scenario.S]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 20, s.Tillering: 50, s.Flowering: 75, s.Senescence: 90}))
    SO = FOCUSCropMixin(display_name= "Soybeans", 
                        defined_scenarios=tuple([Scenario.P]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 35, s.Tillering: 55, s.Flowering: 85, s.Senescence: 65}))
    SW = FOCUSCropMixin(display_name= "Strawberries", 
                        defined_scenarios=tuple([Scenario.H, Scenario.J, Scenario.K, Scenario.S]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 30, s.Tillering: 50, s.Flowering: 60, s.Senescence: 60}))
    TB = FOCUSCropMixin(display_name= "Tobacco", 
                        defined_scenarios=tuple([Scenario.P, Scenario.T]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 50, s.Tillering: 70, s.Flowering: 90, s.Senescence: 90}))
    TM = FOCUSCropMixin(display_name= "Tomatoes", 
                        defined_scenarios=tuple([Scenario.C, Scenario.P, Scenario.O, Scenario.S, Scenario.T]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 50, s.Tillering: 70, s.Flowering: 80, s.Senescence: 50}))
    VI = FOCUSCropMixin(display_name= "Vines", 
                        defined_scenarios=tuple([Scenario.C, Scenario.H, Scenario.K, Scenario.P, Scenario.O, Scenario.S, Scenario.T]),
                        interception=HashableRSDict({s.Maturity: 85, s.DevelopmentFruit: 70, s.Flowering: 60, s.Inflorescence: 50, s.Germination: 40}),
                    )
    WC = FOCUSCropMixin(display_name= "Winter cereals", 
                        defined_scenarios=tuple([Scenario.C, Scenario.H, Scenario.J, Scenario.N, Scenario.P, Scenario.O]),
                        interception=HashableRSDict({s.Germination: 0, s.Leaf: 25, s.Tillering: 50, s.Elongation: 70, s.Flowering: 90, s.Senescence: 90})) # K, S, T have crp files but are not officially defined there

    @staticmethod
    def from_acronym( acronym: str) -> 'FOCUSCrop':
        """Fetches the crop defined by the acronym
        :param acronym: the acronym to fetch
        :return: The crop with the name of the acronym
        >>> crop = FOCUSCrop.from_acronym('VI')
        >>> crop.name == 'VI'
        True
        >>> type(crop)
        <enum 'FOCUSCrop'>
        """
        return FOCUSCrop[acronym]
    
    def get_interception(self, bbch: int) -> float:
        """Gets the interception of this plant for a given development stadium.
        Returns no interception for bbch < 0
        :param bbch: The stadium to check
        :return: The interception for that stadium
        >>> FOCUSCrop.VI.get_interception(80)
        85
        >>> FOCUSCrop.VI.get_interception(50)
        50
        >>> FOCUSCrop.VI.get_interception(20)
        40"""
        t = Timing(bbch)
        if t.principal_stage == PrincipalStage.Unplanted:
            t = replace(t, bbch_state = 0)
        for key, value in self.interception.items():
            if t.principal_stage >= key:
                return value
        raise AssertionError("No fitting interception was defined or bbch was not comparable to float")

@dataclass(frozen=True)
class Timing(TypeCorrecting):
    '''During which bbch stadium will the application happen'''
    bbch_state: int
    '''Relative to which development state'''

    @property
    def principal_stage(self):
        """Returns the principal stage for the timing
        >>> Timing(80).principal_stage == PrincipalStage.Maturity
        True
        >>> Timing(-5).principal_stage == PrincipalStage.Unplanted
        True"""
        return PrincipalStage(max(-1, min(9, math.floor(self.bbch_state / 10))))

@dataclass(frozen=True)
class Application(TypeCorrecting):
    '''Application information'''
    rate: float
    '''How much compound will be applied in g/ha'''
    timing: Timing
    '''What is the timing of application'''
    number: int = 1
    '''How often will be applied'''
    interval: int = 1
    '''What is the minimum interval between applications'''
    factor: float = 1

@dataclass(frozen=True)
class GAP(TypeCorrecting):
    '''Defines a GAP'''
    modelCrop: FOCUSCrop
    '''The crop that the field is modelled after'''
    application: Application
    '''The values of the actual application'''

    def __hash__(self) -> int: # make GAP hash stable
        return hash((self.application, tuple(ord(c) for c in self.modelCrop.name)))

    def _asdict(self):
        """Fixes issues with serialization but relies on a custom JSON Encoder
        >>> import json
        >>> from util.conversions import EnhancedJSONEncoder
        >>> g = GAP("AP", {"rate": 1, "number": 1, "interval": 1, "timing": {"bbch_state": 50}})
        >>> json.dumps(g, cls=EnhancedJSONEncoder)
        '{"modelCrop": "AP", "application": {"rate": 1.0, "timing": {"bbch_state": 50}, "number": 1, "interval": 1, "factor": 1.0}}'
        """
        return {"modelCrop": self.modelCrop.name, "application": asdict(self.application)}

    def from_excel(excel_file: Path) -> List['GAP']:
        gaps = pandas.read_excel(io=excel_file, sheet_name = "GAP Properties")
        return [GAP(
            modelCrop=row['Model Crop'],
            application=Application(
                rate=row['Rate'],
                number=row['Number'],
                interval=row['Interval'],
                timing=Timing(
                    bbch_state=row['BBCH']
                )
            )
        ) for _, row in gaps.iterrows()]