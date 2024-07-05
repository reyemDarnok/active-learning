import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from focusStepsPelmo.ioTypes.compound import Compound
from focusStepsPelmo.ioTypes.gap import GAP, FOCUSCrop, Scenario
from focusStepsPelmo.util.datastructures import TypeCorrecting


@dataclass()
class WaterHorizon:
    """Class describing a horizon segment line of a WASSER.PLM"""
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
        """Expects a single line of the WASSER.PLM from a horizon segment"""
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
        self.temperature = float(segments[0])


@dataclass()
class WaterPLM:
    """A parsed form of a WASSER.PLM file in the Pelmo output. Does not yet parse all fields"""
    horizons: List[List[WaterHorizon]]
    """A 2d List of horizons. Accessed as horizons[year][compartment]"""

    def __init__(self, file: Path) -> None:
        """Expects the path to the WASSER.PLM"""
        years = [[[line for line in section.splitlines()[:-1] if line]
                  for section in re.split(r"---+", year)[1:]]
                 for year in file.read_text().split("ANNUAL WATER OUTPUT")[1:]]
        self.horizons = [[WaterHorizon(line) for line in year[2][:-1]] for year in years]


@dataclass()
class ChemHorizon:
    """Class describing a horizon segment line of a CHEM.PLM"""
    horizon: int
    compartment: int
    soil_application: float
    previous_storage: float
    leaching_input: float
    decay: float
    gas_diffusion: float
    plant_uptake: float
    leaching_output: float
    current_storage: float
    storage_in_neq_domain: float

    def __init__(self, line: str):
        """Expects a single line of the CHEM.PLM from a horizon segment"""
        segments = line.split()
        self.horizon = int(segments[0])
        self.compartment = int(segments[1])
        self.soil_application = float(segments[2])  # might be bool, is written as 1.000/0.0000 value
        self.previous_storage = float(segments[3])
        self.leaching_input = float(segments[4])
        self.decay = float(segments[5])
        self.gas_diffusion = float(segments[6])
        self.plant_uptake = float(segments[7])
        self.leaching_output = float(segments[8])
        self.current_storage = float(segments[9])
        self.storage_in_neq_domain = float(segments[10])


@dataclass()
class ChemPLM:
    """A parsed form of a CHEM.PLM file in the Pelmo output. Does not yet parse all fields"""
    horizons: List[List[ChemHorizon]]
    """A 2d List of horizons. Accessed as horizons[year][compartment]"""

    def __init__(self, file: Path) -> None:
        """Expects the path to a CHEM.PLM file"""
        years = [[[line for line in section.splitlines()[:-1] if line]
                  for section in re.split(r"---+", year)[1:]]
                 for year in file.read_text().split("ANNUAL")[1:]]
        self.horizons = [[ChemHorizon(line) for line in year[2][:-1] if '*' not in line] for year in years]


@dataclass(frozen=True)
class PelmoResult(TypeCorrecting):
    psm_comment: str
    scenario: Scenario
    crop: FOCUSCrop
    pec: Tuple[float, ...]

    def _asdict(self):
        # noinspection PyProtectedMember
        return {"psm_comment": self.psm_comment,
                "scenario": self.scenario.name,
                "crop": self.crop.name,
                "pec": self.pec}


@dataclass(frozen=True)
class PECResult:
    compound: Compound
    gap: GAP
    scenario: Scenario
    crop: FOCUSCrop
    pec: Tuple[float, ...]

    def asdict(self):
        return {"compound": self.compound,
                "gap": self.gap,
                "scenario": self.scenario,
                "crop": self.crop.name,
                "pec": self.pec}
