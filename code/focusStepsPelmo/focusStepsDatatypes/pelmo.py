from dataclasses import dataclass
from pathlib import Path
import re
from typing import List


@dataclass
class WaterHorizon():
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
    horizons: List[List[WaterHorizon]]
    def __init__(self, file: Path) -> None:
        years = [[[line for line in section.splitlines()[:-1] if line] for section in re.split(r"---+", year)[1:]] for year in file.read_text().split("ANNUAL WATER OUTPUT")[1:]]
        self.horizons = [[WaterHorizon(line) for line in year[2][:-1]] for year in years]

@dataclass
class ChemHorizon():
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
    horizons: List[List[ChemHorizon]]
    def __init__(self, file: Path) -> None:
        years = [[[line for line in section.splitlines()[:-1] if line] for section in re.split(r"---+", year)[1:]] for year in file.read_text().split("ANNUAL")[1:]]
        self.horizons = [[ChemHorizon(line) for line in year[2][:-1] if not '*' in line] for year in years]

@dataclass
class PelmoResult:
    pecs: List[float]
    PECgw: float
    def __init__(self, water: WaterPLM, chem: ChemPLM, target_compartment: int = 21) -> None:
        chem_horizons = [horizon for year in chem.horizons for horizon in year if horizon.compartment == target_compartment]
        water_horizons = [horizon for year in water.horizons for horizon in year if horizon.compartment == target_compartment]
        # mass in g/ha / water in mm
        # input is in kg/ha and cm
        self.pecs = [chem_horizons[i].leaching_output * 1000 / (water_horizons[i].leaching_output * 10 ) * 100 for i in range(len(chem_horizons))]
        self.pecs.sort()
        percentile = 0.8
        lower = int((len(self.pecs) - 1) * percentile) + 1
        self.PECgw = (self.pecs[lower] + self.pecs[lower + 1]) / 2

