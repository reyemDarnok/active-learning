

from dataclasses import asdict, dataclass, field
from enum import Enum, auto
import math
from typing import List, Tuple

from util.conversions import map_to_class, str_to_enum
from ioTypes.compound import Compound, Degradation, Sorption
from ioTypes.gap import GAP, Application, FOCUSCrop

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
        else: return Emergence.first_emergence

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

    stage: Emergence = None
    offset: int = None

    def __post_init__(self):
        super().__post_init__()
        if self.stage == None:
            self.stage = Emergence.fromStage(self.timing.principal_stage())
        if self.offset == None:
            if self.timing.bbch_state > 90: self.offset = self.timing.bbch_state - 90
            elif self.timing.bbch_state > 80: self.offset = self.timing.bbch_state - 80
            else: self.offset = self.timing.bbch_state
        



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
    depth_volatility: float = 0.1
    hv: float = 98400
    temperature: float = 20

    
@dataclass
class Moisture:
    '''Used by Pelmo'''
    absolute: float = 0
    relative: float = 100
    exp: float = 0.7

@dataclass
class DegradationData:
    rate: float
    temperature: float = 20
    q10: float = 2.58
    moisture: Moisture = Moisture()
    rel_deg_new_sites: float = 0
    formation_factor: float = 1
    inverse_rate: float = 0
    i_ref: float = 100

    def __post_init__(self):
        self.moisture = map_to_class(self.moisture, Moisture)

@dataclass
class PsmDegradation:
    to_disregard: DegradationData
    metabolites: List['PsmDegradation'] = field(default_factory=list)

    def __post_init__(self):
        self.to_disregard = map_to_class(self.to_disregard, DegradationData)
        if self.metabolites is not None:
            self.metabolites = [map_to_class(x, PsmDegradation) for x in self.metabolites]
            if len(self.metabolites) < 4:
                for _ in range(4-len(self.metabolites)):
                    next_filler = PsmDegradation(to_disregard=DegradationData(rate=0), metabolites=None)
                    self.metabolites += [next_filler]
        else:
            self.metabolites = []



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
    degradations: PsmDegradation # rate calculation with metabolites is still suspect - works for parent only
    adsorptions: List[PsmAdsorption]
    crop: FOCUSCrop
    molar_mass: float
    comment: str = "No comment"
    num_soil_horizons: int = 0
    do_henry_calc: bool = True
    do_kd_calc: bool = True
    volatizations: List[Volatization] = field(default_factory=lambda: [Volatization(henry=3.33E-04, solubility=90, vaporization_pressure=1.00E-04), Volatization(henry=6.67E-10, solubility=180, vaporization_pressure=4.00E-04)])
    plant_uptake: float = 0.5
    degradation_type: DegradationType = DegradationType.FACTORS

    def _asdict(self):
        return {
            "application": self.application,
            "degradations": self.degradations,
            "adsorptions": self.adsorptions,
            "crop": self.crop,
            "molar_mass": self.molar_mass,
            "comment": self.comment,
            "num_soil_horizons": self.num_soil_horizons,
            "do_henry_calc": self.do_henry_calc,
            "do_kd_calc": self.do_kd_calc,
            "volatizations": self.volatizations,
            "plant_uptake": self.plant_uptake,
            "degradation_type": self.degradation_type
            }
    
    @staticmethod
    def fromInput(compound: Compound, gap: GAP) -> 'PsmFile':
        application = PsmApplication(**asdict(gap.application))
        
        degradations = PsmDegradation(to_disregard= DegradationData(rate=math.log(2) / compound.degradation.system), metabolites=[])
        
        adsorptions = [PsmAdsorption(koc = compound.sorption.koc, freundlich=compound.sorption.freundlich)]
        crop = gap.modelCrop
        molar_mass = compound.molarMass
        return PsmFile(application=application, 
                       degradations=degradations, 
                       adsorptions=adsorptions, 
                       crop=crop, molar_mass=molar_mass,
                       plant_uptake=compound.plant_uptake
        )

    def toInput(self) -> Tuple[Compound, GAP]:
        compound = Compound(molarMass=self.molar_mass,
                            waterSolubility=self.volatizations[0].solubility,
                            sorption=Sorption(koc=self.adsorptions[0].koc, freundlich=self.adsorptions[0].freundlich),
                            degradation=Degradation(system=math.log(2)/self.degradations[0].rate,
                                                    soil=math.log(2)/self.degradations[0].rate,
                                                    surfaceWater=math.log(2)/self.degradations[0].rate,
                                                    sediment=math.log(2)/self.degradations[0].rate),
                            plant_uptake=self.plant_uptake
                            )
        gap = GAP(modelCrop=self.crop, application=self.application)
        return (compound, gap)

    def __post_init__(self):
        self.application = map_to_class(self.application, PsmApplication)
        self.degradations = map_to_class(self.degradations, PsmDegradation)
        self.adsorptions = [map_to_class(x, PsmAdsorption) for x in self.adsorptions]
        self.crop = str_to_enum(self.crop, FOCUSCrop)
        self.volatizations = [map_to_class(x, Volatization) for x in self.volatizations]
        self.degradation_type = str_to_enum(self.degradation_type, DegradationType)

