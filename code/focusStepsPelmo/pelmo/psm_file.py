

from dataclasses import asdict, dataclass, field
from enum import Enum, auto
import math
from typing import List, Tuple

from inputTypes.compound import Compound, Degradation, Sorption
from inputTypes.gap import GAP, Application, FOCUSCrop
from inputTypes.pelmo import PelmoCrop

PELMO_UNSET = -99

class Emergence(int, Enum):
    '''The possible application crop development timings for Pelmo'''
    first_emergence = 0
    first_maturation = 1
    first_harvest = 2
    second_emregence = 3
    second_maturation = 4
    second_harvest = 5

    @staticmethod
    def fromStage(stage: int) -> 'Emergence':
        if stage >= 9: return Emergence.first_harvest
        if stage >= 8: return Emergence.first_maturation
        if stage >= 0: return Emergence.first_emergence

class ApplicationType(int, Enum):
    """The different types of application Pelmo recognizes"""
    soil = 1
    linear = 2
    exp_foliar = 3
    manual = 4

@dataclass
class PsmApplication(Application):
    type: ApplicationType = ApplicationType.soil
    lower_depth: float = 0
    upper_depth: float = 0
    ffield: float = 0
    frpex: float = 0
    time: float = 0

    @property
    def stage(self) -> Emergence:
        return Emergence.fromStage(self.timing.principal_stage)

    @property
    def offset(self) -> int:
        if self.timing.bbch_state > 90: return self.timing.bbch_state - 90
        if self.timing.bbch_state > 80: return self.timing.bbch_state - 80
        return self.timing.bbch_state



class DegradationType(int, Enum):
    '''Used by Pelmo to describe the type of degdradation'''
    FACTORS = 0
    CONSTANT_WITH_DEPTH = auto()
    INDIVIDUAL = auto()
    FACTORS_LIQUID_PHASE = auto()
    CONSTANT_WITH_DEPTH_LIQUID_PHASE = auto()
    INDIVIDUAL_LIQUID_PHASE = auto()


@dataclass
class Volatization:
    '''Used by Pelmo'''
    henry: float = 3.33E-04
    solubility: float = 90
    vaporization_pressure: float = 1.00E-04
    diff_air: float = 0.0498
    depth_volatility: float = 98400
    hv: float = 98400
    temperature: float = 20

    
@dataclass
class Moisture:
    '''Used by Pelmo'''
    absolute: float = 0
    relative: float = 100
    exp: float = 0.7

@dataclass
class PsmDegradation:
    rate: float
    temperature: float = 20
    q10: float = 2.58
    moisture: Moisture = Moisture()
    rel_deg_new_sites: float = 0
    formation_factor: float = 1
    inverse_rate: float = 0
    i_ref: float = 100

@dataclass
class PsmAdsorption:
    '''Information about the sorption behavior of a compound. Steps12 uses the koc, Pelmo uses all values'''
    koc: float
    freundlich: float
    pH: float = PELMO_UNSET
    pKa: float = 20
    limit_freundl: float = 0
    annual_increment: float = 0
    k_doc: float = 0
    percent_change: float = 100
    koc2: float = PELMO_UNSET
    pH2: float = PELMO_UNSET
    f_neq: float = 0
    kdes: float = 0

@dataclass
class PsmFile:
    application: PsmApplication
    degradations: List[PsmDegradation]
    adsorption: List[PsmAdsorption]
    crop: FOCUSCrop
    molar_mass: float
    comment: str = "No comment"
    num_soil_horizons: int = 0
    do_henry_calc: bool = True
    do_kd_calc: bool = True
    volatizations: List[Volatization] = field(default_factory=lambda: [Volatization()])
    plant_uptake: float = 0.5
    degradation_type: DegradationType = DegradationType.FACTORS

    @staticmethod
    def fromInput(compound: Compound, gap: GAP) -> 'PsmFile':
        application = PsmApplication(**asdict(gap.application))
        degradation = [PsmDegradation(rate = math.log(2) / metabolite.system) for metabolite in compound.degradation.metabolites]
        degradation += [PsmDegradation(rate = math.log(2) / compound.degradation.system)]
        adsorption = [PsmAdsorption(koc = compound.sorption.koc, freundlich=compound.sorption.freundlich)]
        crop = gap.modelCrop
        molar_mass = compound.molarMass
        return PsmFile(application=application, 
                       degradation=degradation, 
                       adsorption=adsorption, 
                       crop=crop, molar_mass=molar_mass,
                       plant_uptake=compound.plant_uptake)

    def toInput(self) -> Tuple[Compound, GAP]:
        compound = Compound(molarMass=self.molar_mass,
                            waterSolubility=self.volatizations[0].solubility,
                            sorption=Sorption(koc=self.adsorption[0].koc, freundlich=self.adsorption[0].freundlich),
                            degradation=Degradation(system=math.log(2)/self.degradations[0].rate,
                                                    soil=math.log(2)/self.degradations[0].rate,
                                                    surfaceWater=math.log(2)/self.degradations[0].rate,
                                                    sediment=math.log(2)/self.degradations[0].rate),
                            plant_uptake=self.plant_uptake
                            )
        gap = GAP(modelCrop=self.crop, application=self.application)
        return (compound, gap)

