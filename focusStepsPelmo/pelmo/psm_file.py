import math
from dataclasses import asdict, dataclass, field, replace
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from focusStepsPelmo.ioTypes.compound import Compound, Degradation, MetaboliteDescription, Sorption, Volatility
from focusStepsPelmo.ioTypes.gap import GAP, Application, FOCUSCrop
from focusStepsPelmo.util.conversions import map_to_class, str_to_enum

PELMO_UNSET = -99


class Emergence(int, Enum):
    """The possible application crop development timings for Pelmo"""
    first_emergence = 0
    first_maturation = 1
    first_harvest = 2
    second_emergence = 3
    second_maturation = 4
    second_harvest = 5

    @staticmethod
    def from_stage(stage: int) -> 'Emergence':
        if stage >= 9:
            return Emergence.first_harvest
        if stage >= 8:
            return Emergence.first_maturation
        else:
            return Emergence.first_emergence


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
    offset: Optional[int] = 0

    @property
    def stage(self) -> Emergence:
        return Emergence.from_stage(self.timing.principal_stage)

    @property
    def rate_in_kg(self):
        return self.rate / 1000

    def __post_init__(self):
        super().__post_init__()
        if self.offset is None:
            if self.timing.bbch_state > 90:
                object.__setattr__(self, 'offset', self.timing.bbch_state - 90)
            elif self.timing.bbch_state > 80:
                object.__setattr__(self, 'offset', self.timing.bbch_state - 80)
            else:
                object.__setattr__(self, 'offset', self.timing.bbch_state)


class DegradationType(int, Enum):
    """Used by Pelmo to describe the type of degradation"""
    FACTORS = 0
    CONSTANT_WITH_DEPTH = auto()
    INDIVIDUAL = auto()
    FACTORS_LIQUID_PHASE = auto()
    CONSTANT_WITH_DEPTH_LIQUID_PHASE = auto()
    INDIVIDUAL_LIQUID_PHASE = auto()


@dataclass
class Volatization:
    """Used by Pelmo"""
    henry: float = 3.33E-04
    solubility: float = 90
    vaporization_pressure: float = 1.00E-04
    diff_air: float = 0.0498
    depth_volatility: float = 0.1
    hv: float = 98400
    temperature: float = 20


@dataclass
class Moisture:
    """Used by Pelmo"""
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
    metabolites: Optional[Tuple['PsmDegradation', ...]] = field(default_factory=tuple)

    # None if it is degradation to BR/CO2

    def __post_init__(self):
        object.__setattr__(self, 'to_disregard', map_to_class(self.to_disregard, DegradationData))
        if self.metabolites is not None:
            if len(self.metabolites) == 4:
                object.__setattr__(self, 'metabolites', [map_to_class(x, PsmDegradation) for x in self.metabolites])
            elif len(self.metabolites) < 4:
                metabolites = list(self.metabolites)
                for _ in range(4 - len(self.metabolites)):
                    next_filler = PsmDegradation(to_disregard=DegradationData(rate=0), metabolites=None)
                    metabolites += [next_filler]
                    object.__setattr__(self, 'metabolites', tuple(metabolites))
        else:
            object.__setattr__(self, 'metabolites', tuple())


@dataclass
class PsmAdsorption:
    """Information about the sorption behavior of a compound. Steps12 uses the koc, Pelmo uses all values"""
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
    adsorptions: Tuple[PsmAdsorption, ...]
    degradations: List[DegradationData]
    volatizations: Tuple[Volatization, Volatization]
    plant_uptake: float = 0.5
    degradation_type: DegradationType = DegradationType.FACTORS
    name: str = "Unknown name"
    position: Optional[str] = None

    @staticmethod
    def from_compound(compound: Compound) -> 'PsmCompound':
        if compound.degradation.soil > 0:
            full_rate = math.log(2) / compound.degradation.soil
        else:
            full_rate = 0
        remaining_degradation_fraction = 1.0
        degradations = []
        if compound.metabolites is not None:
            for met_des in compound.metabolites:
                if met_des:
                    remaining_degradation_fraction -= met_des.formation_fraction
                    degradations += [DegradationData(rate=full_rate * met_des.formation_fraction)]
                else:
                    degradations += [DegradationData(rate=0)]

        assert remaining_degradation_fraction >= 0, "The sum of formation fractions may not exceed 1"
        degradations += [DegradationData(rate=full_rate * remaining_degradation_fraction)]
        volatizations = (Volatization(henry=3.33E-04, solubility=compound.volatility.water_solubility,
                                      vaporization_pressure=compound.volatility.vaporization_pressure,
                                      temperature=compound.volatility.reference_temperature - 1),
                         Volatization(henry=3.33E-04, solubility=compound.volatility.water_solubility,
                                      vaporization_pressure=compound.volatility.vaporization_pressure,
                                      temperature=compound.volatility.reference_temperature + 1))
        if 'pelmo' in compound.model_specific_data.keys():
            position = compound.model_specific_data['pelmo']['position']
        else:
            position = None
        return PsmCompound(molar_mass=compound.molarMass,
                           adsorptions=tuple(
                               [PsmAdsorption(koc=compound.sorption.koc, freundlich=compound.sorption.freundlich)]),
                           plant_uptake=compound.plant_uptake, degradations=degradations, name=compound.name,
                           volatizations=volatizations,
                           position=position)


PsmCompound.empty = PsmCompound(molar_mass=0, adsorptions=tuple([PsmAdsorption(koc=0, freundlich=1)]), degradations=[],
                                volatizations=(Volatization(), Volatization()))


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
    def from_input(compound: Compound, gap: GAP) -> 'PsmFile':
        application = PsmApplication(**asdict(gap.application))

        metabolites: Dict[str, Compound] = {}
        if 'pelmo' in compound.model_specific_data.keys():
            all_metabolites = [met for met in compound.metabolites] + \
                              [met for metabolite in compound.metabolites for met in metabolite.metabolite.metabolites]

            def compound_position(to_find: Compound) -> str:
                return to_find.model_specific_data.get('pelmo', {}).get('position', 'Unknown Position').upper()

            for current in all_metabolites:
                metabolites[compound_position(current.metabolite)] = current.metabolite
        else:
            for index, metabolite in enumerate(compound.metabolites):
                metabolites[chr(ord('A') + index) + "1"] = metabolite.metabolite
                if metabolite.metabolite.metabolites:
                    metabolites[chr(ord('A') + index) + "2"] = metabolite.metabolite.metabolites[0].metabolite

        def find_formation(parent: Compound, metabolite_position: str,
                           default: Optional[MetaboliteDescription] = None
                           ) -> Optional[MetaboliteDescription]:
            if metabolite_position in metabolites.keys():
                return parent.metabolite_description_by_name(metabolites[metabolite_position].name)
            else:
                return default

        for position in ('D2', 'C2', 'B2', 'A2'):
            if position in metabolites.keys():
                follow_position = chr(ord(position[0]) + 1) + position[1]
                follow_formation = find_formation(metabolites[position], follow_position)
                metabolites[position] = replace(metabolites[position], metabolites=(follow_formation,),
                                                model_specific_data={'pelmo': {'position': position}})
        for position in ('D1', 'C1', 'B1', 'A1'):
            if position in metabolites.keys():
                follow_position = chr(ord(position[0]) + 1) + position[1]
                follow_formation = find_formation(metabolites[position], follow_position)
                down_position = position[0] + chr(ord(position[1]) + 1)
                down_formation = find_formation(metabolites[position], down_position)
                diagonal_position = chr(ord(position[0]) + 1) + chr(ord(position[1]) + 1)
                diagonal_formation = find_formation(metabolites[position], diagonal_position)
                metabolites[position] = replace(metabolites[position],
                                                metabolites=(follow_formation, down_formation, diagonal_formation),
                                                model_specific_data={'pelmo': {'position': position}})
        a1_formation = find_formation(compound, 'A1')
        b1_formation = find_formation(compound, 'B1')
        c1_formation = find_formation(compound, 'C1')
        d1_formation = find_formation(compound, 'D1')
        compound = replace(compound, metabolites=(a1_formation, b1_formation, c1_formation, d1_formation))
        psm_compound = PsmCompound.from_compound(compound)
        metabolite_list: List[PsmCompound] = []
        for position in ('A1', 'B1', 'C1', 'D1', 'A2', 'B2', 'C2', 'D2'):
            if position in metabolites.keys():
                metabolite_list.append(PsmCompound.from_compound(metabolites[position]))
        return PsmFile(application=application,
                       compound=psm_compound,
                       metabolites=metabolite_list,
                       crop=gap.modelCrop,
                       comment="No comment",
                       num_soil_horizons=0,
                       degradation_type=DegradationType.FACTORS,
                       )

    def to_input(self) -> Tuple[Compound, GAP]:
        """Convert this psmFile to ioTypes input data.
        WARNING: This is lossy, as psmFiles do not use all data from the input files"""
        volatility = Volatility(water_solubility=self.compound.volatizations[0].solubility,
                                vaporization_pressure=self.compound.volatizations[0].vaporization_pressure,
                                reference_temperature=(self.compound.volatizations[0].temperature +
                                                       self.compound.volatizations[1].temperature) / 2)
        compound = Compound(molarMass=self.compound.molar_mass,
                            volatility=volatility,
                            sorption=Sorption(koc=self.compound.adsorptions[0].koc,
                                              freundlich=self.compound.adsorptions[0].freundlich),
                            degradation=Degradation(system=math.log(2) / self.compound.degradations[0].rate,
                                                    soil=math.log(2) / self.compound.degradations[0].rate,
                                                    surfaceWater=math.log(2) / self.compound.degradations[0].rate,
                                                    sediment=math.log(2) / self.compound.degradations[0].rate),
                            plant_uptake=self.compound.plant_uptake
                            )
        gap = GAP(modelCrop=self.crop, application=self.application)
        return compound, gap

    def __post_init__(self):
        self.application = map_to_class(self.application, PsmApplication)
        self.compound = map_to_class(self.compound, PsmCompound)
        self.metabolites = [map_to_class(metabolite, PsmCompound) if metabolite else PsmCompound.empty for metabolite in
                            self.metabolites]
        self.crop = str_to_enum(self.crop, FOCUSCrop)
        self.num_soil_horizons = int(self.num_soil_horizons)
        self.degradation_type = str_to_enum(self.degradation_type, DegradationType)
