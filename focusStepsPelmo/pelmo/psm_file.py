"""A file describing a psm file and its components"""
import math
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from jinja2 import Environment, select_autoescape, StrictUndefined, PackageLoader, Template

from focusStepsPelmo.ioTypes.compound import Compound, MetaboliteDescription, DT50
from focusStepsPelmo.ioTypes.gap import GAP, GAPMachineGAP, Scenario, FOCUSCrop
from focusStepsPelmo.util.datastructures import TypeCorrecting

PELMO_UNSET = -99
"""A value Pelmo uses for inapplicable values"""
jinja_env = Environment(loader=PackageLoader('focusStepsPelmo.pelmo'),
                        autoescape=select_autoescape(), undefined=StrictUndefined)


def setup_psm_template() -> Template:
    """Initialises the jinja2 template for psm files"""
    global_data = {'dummy_gap': GAPMachineGAP(modelCrop=FOCUSCrop.AP, rate=0, interceptions=tuple([0]),
                                              first_season={Scenario.C: datetime(year=1, month=1, day=1)}),
                   'dummy_scenario': Scenario.C}
    return jinja_env.get_template('general.psm.j2', globals=global_data)


psm_template = setup_psm_template()


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
        """Turn bbch major stages into Pelmo emergence values"""
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
    type: ApplicationType = ApplicationType.manual
    """Which application type was selected"""
    lower_depth: float = 0
    """The lower depth of incorporation in cm (only relevant for soil application)"""
    upper_depth: float = 0
    """"The upper depth of incorporation in cm (only relevant for soil application"""
    ffield: float = 0
    """The rapidly dissipating fraction at the soil surface"""
    frpex: float = 0
    """Fraction of poorly exposed pesticide"""
    time: float = 0
    """Hour of application"""


class DegradationType(int, Enum):
    """Used by Pelmo to describe the type of degradation"""
    FACTORS = 0
    CONSTANT_WITH_DEPTH = auto()
    INDIVIDUAL = auto()
    FACTORS_LIQUID_PHASE = auto()
    CONSTANT_WITH_DEPTH_LIQUID_PHASE = auto()
    INDIVIDUAL_LIQUID_PHASE = auto()


@dataclass(frozen=True)
class Volatization:
    """Used by Pelmo"""
    henry: float = 3.33E-04
    """Henry constant in J/mol"""
    solubility: float = 90
    """Water solubility in mg/L"""
    vaporization_pressure: float = 1.00E-04  # TODO rename
    """Vapor pressure in Pa"""
    diff_air: float = 0.0498
    """Diffusion Coefficient Air at 20°C in cm^2/s"""
    depth_volatility: float = 0.1
    """Thickness of Boundary layer in cm"""
    hv: float = 98400  # TODO document
    temperature: float = 20
    """Reference temperature in °C"""


@dataclass(frozen=True)
class Moisture:
    """Used by Pelmo"""
    absolute: float = 0
    """Absolute moisture during study in Volume %"""
    relative: float = 100
    """Relative moisture during study in %FC"""
    exp: float = 0.7
    """Moisture exponent"""


@dataclass(frozen=True)
class DegradationData(TypeCorrecting):
    rate: float
    """Degradation rate, calculated as ln(2)/dt50 in days^-1"""
    temperature: float = 20
    """Reference temperature in °C"""
    q10: float = 2.58
    """Temperature correction factor"""
    moisture: Moisture = Moisture()
    """Moisture information"""
    rel_deg_new_sites: float = 0  # TOD rename
    """Relative degradation at non-equilibrium sites"""
    formation_factor: float = 1
    """stoichiometric factor"""
    photodegradation: float = 0
    """Photolysis degradation in days"""
    reference_irradiance: float = 100
    """Reference irradiance for Photolysis in W/m^2"""
    target: str = "Unknown"
    """Which Pelmo position this degradation points to"""

    def __post_init__(self):
        if self.rate == 0:
            object.__setattr__(self, 'reference_irradiance', 0)


@dataclass(frozen=True)
class PsmAdsorption:
    """Information about the sorption behavior of a compound. Steps12 uses the koc, Pelmo uses all values"""
    koc: float
    """KOC value in L/kg"""
    freundlich: float
    """Freundlich exponent fo the koc value"""
    pH: float = PELMO_UNSET
    """pH value at which the first sorption study was performed (only relevant with pH dependent sorption)"""
    pKa: float = 20
    """pKa value of the compound (only relevant with pH dependent sorption"""
    limit_freundl: float = 0
    """Concentration where Freundlich sorption switches to linear sorption"""
    annual_increment: float = 0
    """Annual decrease of sorption constant (linearly, in percent)"""
    k_doc: float = 0
    """complexation constant to Doc. Unitless. Only relevant if Doc content in soil is > 0"""
    percent_change: float = 100
    """Relative increase of sorption of soil is air dried. Unitless, in percent"""
    koc2: float = PELMO_UNSET
    """Koc value of the compound at pH2 during the second sorption study (only relevant with pH dependent sorption) """
    pH2: float = PELMO_UNSET
    """pH value at which the second sorption study was performed (only relevant with pH dependent sorption)"""
    f_neq: float = 0
    """soil fraction of the non-equilibrium domain. """
    kdes: float = 0
    """1st order desorption rate at non-equilibrium sites"""


@dataclass(frozen=True)
class FateOnCrop:
    """Data about what happens to the substance on the crop"""
    plant_decay_rate: float = 0.0693
    """Decay rate on the plant foliate days^-1 (ln(2)/dt50)"""
    washoff_parameter: float = 0.0
    """Foliar extraction coefficient for substance washoff per cm of precipitation in cm^-1"""
    penetration: float = 0.0693
    """Filtration parameter. Only required in exponential application model. in days^-1"""
    photodegradation: float = 0
    """Photodegradation rate in days^-1 (ln(2)/dt50)"""
    reference_irradiance: float = 100
    """Reference radiation for photodegradation in W/m^2"""
    laminar_layer: float = 0.03
    """Laminar layer for volatilisation from foliate in W/m^2"""


@dataclass(frozen=True)
class PsmCompound:
    """Describes a compound as it appears in a psm file"""
    molar_mass: float
    """molar mass in g/mol"""
    adsorption: PsmAdsorption
    degradations: List[DegradationData]
    volatizations: Tuple[Volatization, Volatization]  # TODO rename volatilization
    plant_uptake: float = 0.5
    degradation_type: DegradationType = DegradationType.FACTORS
    name: str = "Unknown name"
    position: Optional[str] = None
    fate_on_crop: FateOnCrop = FateOnCrop()

    @staticmethod
    def from_compound(compound: Compound) -> 'PsmCompound':
        """Creates a psmCompound from a general Compound"""
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
                    target = "Met " + met_des.metabolite.model_specific_data.get('pelmo', {}).get('position', 'Unknown')
                    degradations += [DegradationData(rate=full_rate * met_des.formation_fraction, target=target)]
                else:
                    degradations += [DegradationData(rate=0, target="Undefined")]

        assert remaining_degradation_fraction >= 0, "The sum of formation fractions may not exceed 1"
        degradations += [DegradationData(rate=full_rate * remaining_degradation_fraction, target="BR/CO2")]
        volatizations = expand_volatilization_regulatory(Volatization(solubility=compound.water_solubility,
                                                                      vaporization_pressure=compound.vapor_pressure,
                                                                      temperature=compound.reference_temperature))

        position = compound.model_specific_data.get('pelmo', {}).get('position')
        return PsmCompound(molar_mass=compound.molarMass,
                           adsorption=PsmAdsorption(koc=compound.koc, freundlich=compound.freundlich),
                           plant_uptake=compound.plant_uptake, degradations=degradations, name=compound.name,
                           volatizations=volatizations,
                           position=position)


PsmCompound.empty = PsmCompound(molar_mass=0, adsorption=PsmAdsorption(koc=0, freundlich=1), degradations=[],
                                volatizations=(Volatization(), Volatization()))


def expand_volatilization_regulatory(volatilization: Volatization) -> Tuple[Volatization, Volatization]:
    """Expand a single volatilization line into two according to regulatory practices"""
    return (replace(volatilization, temperature=volatilization.temperature - 0.5),
            replace(volatilization, temperature=volatilization.temperature + 0.5))


def expand_volatilization_user_manual(volatilization: Volatization) -> Tuple[Volatization, Volatization]:
    """Expand a single volatilization line into two according to the pelmo user manual"""
    return volatilization, replace(volatilization, solubility=volatilization.solubility * 2,
                                   vaporization_pressure=volatilization.vaporization_pressure * 4)


@dataclass(frozen=True)
class PsmFile(TypeCorrecting):
    """Describes the contents of a psm file"""
    application: PsmApplication
    gap: GAP
    compound: PsmCompound
    metabolites: List[PsmCompound]
    comment: str = "No comment"
    num_soil_horizons: int = 0
    degradation_type: DegradationType = DegradationType.FACTORS

    def asdict(self):
        """Represent self as a dict. Necessary because dataclasses.asdict chokes on named tuples in gap"""
        return {
            "gap": self.gap,
            "application": self.application,
            "compound": self.compound,
            "metabolites": self.metabolites,
            "comment": self.comment,
            "num_soil_horizons": self.num_soil_horizons,
            "degradation_type": self.degradation_type,
        }

    @staticmethod
    def from_input(compound: Compound, gap: GAP) -> 'PsmFile':
        """Create a PsmFile from the ioTypes classes"""
        application = PsmApplication()

        metabolites: Dict[str, Compound] = {}
        if 'pelmo' in compound.model_specific_data.keys():
            all_metabolites = [met for met in compound.metabolites] + \
                              [met for metabolite in compound.metabolites for met in metabolite.metabolite.metabolites]

            def compound_position(to_find: Compound) -> str:
                """Find the metabolite position of a compound, i.e. "A1"
                >>> c = Compound(model_specific_data={'pelmo': {'position': "B2"}})
                >>> compound_position(c)
                'B2'
                """
                return to_find.model_specific_data.get('pelmo', {}).get('position', 'Unknown Position').upper()

            for current in all_metabolites:
                position = compound_position(current.metabolite)
                metabolite = current.metabolite
                metabolites[position] = metabolite
        elif compound.metabolites:
            for index, met_des in enumerate(compound.metabolites):
                position = chr(ord('A') + index) + "1"
                all_model_data = met_des.metabolite.model_specific_data.copy()
                model_data = all_model_data.get('pelmo', {})
                model_data['position'] = position
                all_model_data['pelmo'] = model_data
                metabolite = replace(met_des.metabolite, model_specific_data=all_model_data)
                metabolites[position] = metabolite
                if met_des.metabolite.metabolites:
                    position = chr(ord('A') + index) + "2"
                    metabolite = met_des.metabolite.metabolites[0].metabolite
                    metabolites[position] = metabolite

        compound = PsmFile.reorder_metabolites(compound, metabolites)
        psm_compound = PsmCompound.from_compound(compound)
        metabolite_list: List[PsmCompound] = []
        for position in ('A1', 'B1', 'C1', 'D1', 'A2', 'B2', 'C2', 'D2'):
            if position in metabolites.keys():
                met_des = PsmCompound.from_compound(metabolites[position])
                met_adsorption = replace(met_des.adsorption, limit_freundl=1e-20, percent_change=1000)
                met_des = replace(met_des, adsorption=met_adsorption)
                metabolite_list.append(met_des)
        return PsmFile(application=application, compound=psm_compound, metabolites=metabolite_list, gap=gap)

    @staticmethod
    def reorder_metabolites(compound: Compound, metabolites: Dict[str, Compound]) -> Compound:
        """Given a compound and all its metabolites and their Pelmo positions, return a Compound with the degradation
        tree aligned for Pelmo
        :param compound: The parent compound
        :param metabolites: A mapping from the pelmo position of a metabolite to the metabolite
        :return: A Compound that has the same properties as compound, but has its metabolites in Pelmos ordering"""

        def find_formation(parent: Compound, metabolite_position: str,
                           default: Optional[MetaboliteDescription] = None
                           ) -> Optional[MetaboliteDescription]:
            """Find the Metabolite Description for a given metabolite position
            :param parent: The parent compound for the formation
            :param metabolite_position: The position to find
            :param default: What to return when no formation could be found
            :return: The metabolite description for the metabolite_position"""
            if metabolite_position in metabolites.keys():
                return replace(parent.metabolite_description_by_name(metabolites[metabolite_position].name),
                               metabolite=metabolites[metabolite_position])
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
                            vapor_pressure=self.compound.volatizations[0].vaporization_pressure,
                            reference_temperature=self.compound.volatizations[0].temperature,
                            koc=self.compound.adsorption.koc,
                            freundlich=self.compound.adsorption.freundlich,
                            dt50=DT50(system=math.log(2) / self.compound.degradations[0].rate,
                                      soil=math.log(2) / self.compound.degradations[0].rate,
                                      surfaceWater=math.log(2) / self.compound.degradations[0].rate,
                                      sediment=math.log(2) / self.compound.degradations[0].rate),
                            plant_uptake=self.compound.plant_uptake
                            )
        return compound, self.gap

    def render(self) -> str:
        """Render this psm file as a string"""
        template_data = self.asdict()
        rendered = psm_template.render(template_data)
        return rendered
