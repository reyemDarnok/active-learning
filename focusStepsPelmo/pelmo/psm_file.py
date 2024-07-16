import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from jinja2 import Environment, select_autoescape, StrictUndefined, PackageLoader

from focusStepsPelmo.ioTypes.compound import Compound, MetaboliteDescription, DT50
from focusStepsPelmo.ioTypes.gap import GAP
from focusStepsPelmo.util.datastructures import TypeCorrecting

PELMO_UNSET = -99
jinja_env = Environment(loader=PackageLoader('focusStepsPelmo.pelmo'),
                        autoescape=select_autoescape(), undefined=StrictUndefined)


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
class PsmApplication(TypeCorrecting):
    type: ApplicationType = ApplicationType.soil
    lower_depth: float = 0
    upper_depth: float = 0
    ffield: float = 0
    frpex: float = 0
    time: float = 0


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
class DegradationData(TypeCorrecting):
    rate: float
    temperature: float = 20
    q10: float = 2.58
    moisture: Moisture = Moisture()
    rel_deg_new_sites: float = 0
    formation_factor: float = 1
    inverse_rate: float = 0
    i_ref: float = 100


@dataclass
class PsmDegradation(TypeCorrecting):
    to_disregard: DegradationData
    metabolites: Optional[Tuple['PsmDegradation', ...]] = field(default_factory=tuple)

    # None if it is dt50 to BR/CO2

    def __post_init__(self):
        if self.metabolites is not None:
            object.__setattr__(self, 'metabolites', self.metabolites +
                               tuple([PsmDegradation(to_disregard=DegradationData(rate=0), metabolites=None)] *
                                     (4 - len(self.metabolites))))
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
        if compound.dt50.soil > 0:
            full_rate = math.log(2) / compound.dt50.soil
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
        volatizations = (Volatization(henry=3.33E-04, solubility=compound.water_solubility,
                                      vaporization_pressure=compound.vaporization_pressure,
                                      temperature=compound.reference_temperature),
                         Volatization(henry=3.33E-04 * 2, solubility=compound.water_solubility,
                                      vaporization_pressure=compound.vaporization_pressure * 4,
                                      temperature=compound.reference_temperature + 10))
        if 'pelmo' in compound.model_specific_data.keys():
            position = compound.model_specific_data['pelmo']['position']
        else:
            position = None
        return PsmCompound(molar_mass=compound.molarMass,
                           adsorptions=tuple(
                               [PsmAdsorption(koc=compound.koc, freundlich=compound.freundlich)]),
                           plant_uptake=compound.plant_uptake, degradations=degradations, name=compound.name,
                           volatizations=volatizations,
                           position=position)


PsmCompound.empty = PsmCompound(molar_mass=0, adsorptions=tuple([PsmAdsorption(koc=0, freundlich=1)]), degradations=[],
                                volatizations=(Volatization(), Volatization()))


@dataclass
class PsmFile(TypeCorrecting):
    application: PsmApplication
    gap: GAP
    compound: PsmCompound
    metabolites: List[PsmCompound]
    comment: str = "No comment"
    num_soil_horizons: int = 0
    degradation_type: DegradationType = DegradationType.FACTORS

    def asdict(self):
        return {
            "application": self.application,
            "compound": self.compound,
            "metabolites": self.metabolites,
            "comment": self.comment,
            "num_soil_horizons": self.num_soil_horizons,
            "degradation_type": self.degradation_type,
        }

    @staticmethod
    def from_input(compound: Compound, gap: GAP) -> 'PsmFile':
        application = PsmApplication()

        metabolites: Dict[str, Compound] = {}
        if 'pelmo' in compound.model_specific_data.keys():
            all_metabolites = [met for met in compound.metabolites] + \
                              [met for metabolite in compound.metabolites for met in metabolite.metabolite.metabolites]

            def compound_position(to_find: Compound) -> str:
                return to_find.model_specific_data.get('pelmo', {}).get('position', 'Unknown Position').upper()

            for current in all_metabolites:
                metabolites[compound_position(current.metabolite)] = current.metabolite
        elif compound.metabolites:
            for index, metabolite in enumerate(compound.metabolites):
                metabolites[chr(ord('A') + index) + "1"] = metabolite.metabolite
                if metabolite.metabolite.metabolites:
                    metabolites[chr(ord('A') + index) + "2"] = metabolite.metabolite.metabolites[0].metabolite

        compound = PsmFile.reorder_metabolites(compound, metabolites)
        psm_compound = PsmCompound.from_compound(compound)
        metabolite_list: List[PsmCompound] = []
        for position in ('A1', 'B1', 'C1', 'D1', 'A2', 'B2', 'C2', 'D2'):
            if position in metabolites.keys():
                metabolite_list.append(PsmCompound.from_compound(metabolites[position]))
        return PsmFile(application=application,
                       compound=psm_compound,
                       metabolites=metabolite_list,
                       comment="No comment",
                       num_soil_horizons=0,
                       degradation_type=DegradationType.FACTORS,
                       gap=gap
                       )

    @staticmethod
    def reorder_metabolites(compound, metabolites):
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
        return compound

    def to_input(self) -> Tuple[Compound, GAP]:
        """Convert this psmFile to ioTypes input data.
        WARNING: This is lossy, as psmFiles do not use all data from the input files"""
        compound = Compound(molarMass=self.compound.molar_mass,
                            water_solubility=self.compound.volatizations[0].solubility,
                            vaporization_pressure=self.compound.volatizations[0].vaporization_pressure,
                            reference_temperature=self.compound.volatizations[0].temperature,
                            koc=self.compound.adsorptions[0].koc,
                            freundlich=self.compound.adsorptions[0].freundlich,
                            dt50=DT50(system=math.log(2) / self.compound.degradations[0].rate,
                                      soil=math.log(2) / self.compound.degradations[0].rate,
                                      surfaceWater=math.log(2) / self.compound.degradations[0].rate,
                                      sediment=math.log(2) / self.compound.degradations[0].rate),
                            plant_uptake=self.compound.plant_uptake
                            )
        return compound, self.gap

    def render(self) -> str:
        psm_template = jinja_env.get_template('general.psm.j2')
        template_data = self.asdict()
        template_data['dummy_event'] = tuple([datetime(year=1, month=1, day=1), 0])
        template_data['gap'] = self.gap
        rendered = psm_template.render(**template_data)
        return rendered
