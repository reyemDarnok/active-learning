
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict


class Coverage(int, Enum):
    '''Coverage categories for Steps12'''
    No = 0
    Minimal = auto()
    Average = auto()
    Full = auto()

class Region(int, Enum):
    '''The Steps12 Regions'''
    NoRunoff = 0
    North = 1
    South = 2


class Season(int, Enum):
    '''The Step12 Seasons'''
    Spring = auto()
    Summer = auto()
    Autumn = auto()

@dataclass(frozen=True)
class Scenario:
    '''General Scenario information, the defaults do not typically need to be overridden'''
    waterDepth: float = 30
    runOffEventDay: int = 0
    density: float = 0.8
    oc: float = 5
    sedimentDepth: float = 1
    effectiveSedimentDepth: float = 1
    step1RunoffPercentage: float = 10
    fieldToWaterRateio: float = 10
    equilibrationFactor: float = 1.5
    runoffMap: Dict[Region, Dict[Season, float]] = field(default_factory=lambda : {
        Region.NoRunoff: {Season.Spring: 2, Season.Summer: 2, Season.Autumn: 5},
        Region.North: {Season.Spring: 4, Season.Summer: 3, Season.Autumn: 4},
        Region.South: {Season.Spring: 0, Season.Summer: 0, Season.Autumn: 0}
    })


# Compound starts here 
@dataclass(frozen=True)
class Occurrence:
    '''Used for Step12 calculations
    Describes the Occurrence behaviour of a substance.'''
    sediment: float
    '''The Occurrence in sediment. Is always 100 for parents and less for metabolites'''
    soil: float
    '''The Occurrence in soil. Is always 100 for parents and less for metabolites'''