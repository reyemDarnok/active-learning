from dataclasses import dataclass, field
from typing import Dict

from .gap import Region, Season

@dataclass
class Scenario:
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