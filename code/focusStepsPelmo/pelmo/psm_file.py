

from dataclasses import asdict, dataclass, field, replace
from enum import Enum, auto
import math
from typing import Dict, Generator, List, Tuple

from util.conversions import map_to_class, str_to_enum
from ioTypes.compound import Compound, Degradation, MetaboliteDescription, Sorption, Volatility
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

@dataclass(frozen=True)
class PsmApplication(Application):
    type: ApplicationType = ApplicationType.soil
    lower_depth: float = 0
    upper_depth: float = 0
    ffield: float = 0
    frpex: float = 0
    time: float = 0

    stage: Emergence = None
    offset: int = None

    @property
    def rate_in_kg(self):
        return self.rate / 1000

    def __post_init__(self):
        super().__post_init__()
        if self.stage == None:
            object.__setattr__(self, 'stage', Emergence.fromStage(self.timing.principal_stage))
        if self.offset == None:
            if self.timing.bbch_state > 90: object.__setattr__(self, 'offset', self.timing.bbch_state - 90)
            elif self.timing.bbch_state > 80: object.__setattr__(self, 'offset', self.timing.bbch_state - 80)
            else: object.__setattr__(self, 'offset', self.timing.bbch_state)
        



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
        object.__setattr__(self, 'moisture', map_to_class(self.moisture, Moisture))

@dataclass
class PsmDegradation:
    to_disregard: DegradationData
    metabolites: Tuple['PsmDegradation'] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, 'to_disregard', map_to_class(self.to_disregard, DegradationData))
        if self.metabolites is not None:
            if len(self.metabolites) == 4:
                object.__setattr__(self, 'metabolites', [map_to_class(x, PsmDegradation) for x in self.metabolites])
            elif len(self.metabolites) < 4:
                metabolites = list(self.metabolites)
                for _ in range(4-len(self.metabolites)):
                    next_filler = PsmDegradation(to_disregard=DegradationData(rate=0), metabolites=None)
                    metabolites += [next_filler]
                    object.__setattr__(self, 'metabolites', tuple(metabolites))
        else:
            object.__setattr__(self, 'metabolites', tuple())



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
class PsmCompound:
    molar_mass: float
    adsorptions: Tuple[PsmAdsorption]
    degradations: List[DegradationData]
    volatizations: Tuple[Volatization]
    plant_uptake: float = 0.5
    degradation_type: DegradationType = DegradationType.FACTORS
    name: str = "Unknown name"

    @staticmethod
    def from_compound(compound: Compound) -> 'PsmCompound':
        if compound.degradation.soil > 0:
            full_rate = math.log(2)/compound.degradation.soil
        else:
            full_rate = 0
        remaining_degradation_fraction = 1.0
        degradations = []
        for met_des in compound.metabolites:
            remaining_degradation_fraction -= met_des.formation_fraction
            degradations += [DegradationData(rate=full_rate*met_des.formation_fraction)]
        assert remaining_degradation_fraction >= 0, "The sum of formation fractions may not exceed 1"
        if compound.name.lower() in ('a1', 'b1', 'c1', 'd1'):
            missing_metabolites = 3 - len(degradations)
        elif compound.name.lower() in ('a2', 'b2', 'c2', 'd2'):
            missing_metabolites = 1 - len(degradations)
        else:
            missing_metabolites = 4 - len(degradations)
        degradations += [DegradationData(rate=0.0)] * missing_metabolites
        degradations += [DegradationData(rate=full_rate*remaining_degradation_fraction)]
        volatizations = tuple([Volatization(henry=3.33E-04, solubility=compound.volatility.water_solubility, vaporization_pressure=compound.volatility.vaporization_pressure, temperature=compound.volatility.reference_temperature - 1),
                              Volatization(henry=3.33E-04, solubility=compound.volatility.water_solubility, vaporization_pressure=compound.volatility.vaporization_pressure, temperature=compound.volatility.reference_temperature + 1)])
        return PsmCompound(molar_mass=compound.molarMass, 
                    adsorptions=tuple([PsmAdsorption(koc = compound.sorption.koc, freundlich=compound.sorption.freundlich)]),
                    plant_uptake=compound.plant_uptake,degradations=degradations, name=compound.name, volatizations=volatizations)
PsmCompound.empty = PsmCompound(molar_mass=0, adsorptions=tuple([PsmAdsorption(koc = 0, freundlich=1)]),degradations=[],volatizations=tuple())

@dataclass
class PsmFile:
    application: PsmApplication
    compound: PsmCompound
    metabolites: List[PsmCompound]
    crop: FOCUSCrop
    comment: str = "No comment"
    num_soil_horizons: int = 0
    degradation_type: DegradationType = DegradationType.FACTORS

    def _asdict(self):
        return {
            "application": self.application,
            "compound": self.compound,
            "crop": self.crop,
            "comment": self.comment,
            "num_soil_horizons": self.num_soil_horizons,
            "metabolites": self.metabolites,
            "degradation_type": self.degradation_type
            }
    
    @staticmethod
    def fromInput(compound: Compound, gap: GAP) -> 'PsmFile':
        """Convert ioTypes input data to the pelmo specific PsmFile"""
        application = PsmApplication(**asdict(gap.application))
    
        if 'pelmo' in compound.model_specific_data.keys():
            all_metabolites = [met for met in compound.metabolites] + \
                            [met for metabolite in compound.metabolites for met in metabolite.metabolite.metabolites]
            def compound_position(c: Compound) -> str:
                return c.model_specific_data.get('pelmo', {}).get('position', 'Unknown Position').casefold()
            compound_positions: Dict[str, MetaboliteDescription] = {}
            for c in all_metabolites:
                compound_positions[compound_position(c.metabolite)] = c
            a1 = compound_positions('a1', None)
            b1 = compound_positions('b1', None)
            c1 = compound_positions('c1', None)
            d1 = compound_positions('d1', None)
            a2 = compound_positions('a2', None)
            b2 = compound_positions('b2', None)
            c2 = compound_positions('c2', None)
            d2 = compound_positions('d2', None)
        else:
            a1 = compound.metabolites[0] if 0 < len(compound.metabolites) else None
            b1 = compound.metabolites[1] if 1 < len(compound.metabolites) else None
            c1 = compound.metabolites[2] if 2 < len(compound.metabolites) else None
            d1 = compound.metabolites[3] if 3 < len(compound.metabolites) else None
            a2 = a1.metabolite.metabolites[0] if a1 and a1.metabolite.metabolites else None
            b2 = b1.metabolite.metabolites[0] if b1 and b1.metabolite.metabolites else None
            c2 = c1.metabolite.metabolites[0] if c1 and c1.metabolite.metabolites else None
            d2 = d1.metabolite.metabolites[0] if d1 and d1.metabolite.metabolites else None

        if d2:
            d2 = replace(d2, metabolite = replace(d2.metabolite, metabolites=[None], name='D2'))
        if c2:
            c2 = replace(c2, metabolite = replace(c2.metabolite, metabolites=[d2], name='C2'))
        if b2:
            b2 = replace(b2, metabolite = replace(b2.metabolite, metabolites=[c2], name='B2'))
        if a2:
            a2 = replace(a2, metabolite = replace(a2.metabolite, metabolites=[b2], name='A2'))
        if d1:
            d1 = replace(d1, metabolite = replace(d1.metabolite, metabolites=[None, d2, None], name='D1'))
        if c1:
            c1 = replace(c1, metabolite = replace(c1.metabolite, metabolites=[d1, c2, d2], name='C1'))
        if b1:
            b1 = replace(b1, metabolite = replace(b1.metabolite, metabolites=[c1, b2, c2], name='B1'))
        if a1:
            a1 = replace(a1, metabolite = replace(a1.metabolite, metabolites=[b1, a2, b2], name='A1'))
        compound = replace(compound, metabolites=[a1, b1, c1, d1])
        def maybe_from_compound(met_des: MetaboliteDescription) -> PsmCompound:
            return PsmCompound.from_compound(met_des.metabolite) if met_des else None
        psmCompound = PsmCompound.from_compound(compound)
        a1 = maybe_from_compound(a1)
        b1 = maybe_from_compound(b1)
        c1 = maybe_from_compound(c1)
        d1 = maybe_from_compound(d1)
        a2 = maybe_from_compound(a2)
        b2 = maybe_from_compound(b2)
        c2 = maybe_from_compound(c2)
        d2 = maybe_from_compound(d2)

        metabolites = [a1, b1, c2, d2, a2, b2, c2, d2]
        metabolites = list(filter(None, metabolites))
        return PsmFile(application=application,
                       compound=psmCompound,
                       metabolites=metabolites,
                       crop=gap.modelCrop,
                       comment="No comment",
                       num_soil_horizons=0,
                       degradation_type=DegradationType.FACTORS,
        )

    def toInput(self) -> Tuple[Compound, GAP]:
        """Convert this psmFile to ioTypes input data.
        WARNING: This is lossy, as psmFiles do not use all data from the input files"""
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
        self.compound = map_to_class(self.compound, PsmCompound)
        self.metabolites = [map_to_class(metabolite, PsmCompound) if metabolite else PsmCompound.empty for metabolite in self.metabolites]
        self.crop = str_to_enum(self.crop, FOCUSCrop)
        self.num_soil_horizons = int(self.num_soil_horizons)
        self.degradation_type = str_to_enum(self.degradation_type, DegradationType)

