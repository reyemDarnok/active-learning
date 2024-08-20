"""A file describing GAPs and their components"""
import csv
import json
from abc import ABC, abstractmethod
from collections import UserList
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Dict, Generator, List, Tuple, NamedTuple, Any, OrderedDict, FrozenSet, Optional, Union

import pandas

from focusStepsPelmo.util.conversions import excel_date_to_datetime, uncomment
from focusStepsPelmo.util.datastructures import HashableRSDict, TypeCorrecting, correct_type, HashableDict

bbch_application: pandas.DataFrame = pandas.read_csv(Path(__file__).parent / 'BBCHGW.csv',
                                                     header=0,
                                                     dtype={'Location': 'category', 'Crop': 'category',
                                                            'Requested BBCH Code': 'byte',
                                                            'Allocated BBCH Code': 'byte',
                                                            'Crop Interception(%)': 'byte'},
                                                     parse_dates=['Recommended Application date'],
                                                     dayfirst=True)
"""A dataframe containing information about when should the application happen and what is the crop interception
during the application"""


class Scenario(str, Enum):
    """The Pelmo Scenarios. The name is the one letter shorthand and the value the full name"""
    C = "Châteaudun"
    H = "Hamburg"
    J = "Jokioinen"
    K = "Kremsmünster"
    N = "Okehampton"
    P = "Piacenza"
    O = "Porto"
    S = "Sevilla"
    T = "Thiva"


class PrincipalStage(int, Enum):
    """The principal BBCH stages. The name is what the stage is and the value the BBCH Stage / 10 where it is reached"""
    Unplanted = -1
    Germination = 0
    Leaf = 1
    Tillering = 2
    Elongation = 3
    Bolting = 4
    Inflorescence = 5
    Flowering = 6
    DevelopmentFruit = 7
    Maturity = 8
    Senescence = 9


class FOCUSCropMixin(NamedTuple):
    """Crop information for Pelmo"""
    focus_name: str
    """The full name of the focus crop"""
    defined_scenarios: FrozenSet[Scenario]
    """The scenarios that are defined for this Crop in Pelmo"""
    interception: OrderedDict[PrincipalStage, float]
    """Mapping bbch states to interception values in percent. Should be between 0 and 100"""
    bbch_application_name: List[str]
    """The category in the bbch_application data this crop belongs to"""
    alternative_names: List[str] = []
    """Any alternate names for this crop that should also be recognized"""


_s = PrincipalStage


# noinspection PyTypeChecker
class FOCUSCrop(FOCUSCropMixin, Enum):
    """The crops defined for FOCUS. Each defined as a FOCUSCropMixin"""
    AP = FOCUSCropMixin(focus_name="Apples",
                        defined_scenarios=frozenset({
                            Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O,
                            Scenario.S, Scenario.T}),
                        interception=HashableRSDict(
                            {_s.Maturity: 80, _s.DevelopmentFruit: 70, _s.Flowering: 65, _s.Germination: 50})
                        , bbch_application_name=["pome fruit"])
    BB = FOCUSCropMixin(focus_name="Bush berries",
                        defined_scenarios=frozenset({Scenario.J}),
                        interception=HashableRSDict(
                            {_s.Maturity: 80, _s.DevelopmentFruit: 65, _s.Flowering: 65, _s.Germination: 50})
                        , bbch_application_name=["Missing"])  # TODO name for bbch lookup
    BF = FOCUSCropMixin(focus_name="Beans (field)",
                        defined_scenarios=frozenset({Scenario.H, Scenario.K, Scenario.N}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 25, _s.Tillering: 40, _s.Flowering: 70, _s.Senescence: 80})
                        , bbch_application_name=["beans 1", "beans 2"])
    BV = FOCUSCropMixin(focus_name="Beans (vegetables)",
                        defined_scenarios=frozenset({Scenario.O, Scenario.T}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 25, _s.Tillering: 40, _s.Flowering: 70, _s.Senescence: 80})
                        , bbch_application_name=["beans 1", "beans 2"])
    CA = FOCUSCropMixin(focus_name="Carrots",
                        defined_scenarios=frozenset({Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.O,
                                                     Scenario.T}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 25, _s.Tillering: 60, _s.Flowering: 80, _s.Senescence: 80})
                        , bbch_application_name=["carrots 1", "carrots 2"])
    CB = FOCUSCropMixin(focus_name="Cabbage",
                        defined_scenarios=frozenset({Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.O,
                                                     Scenario.S, Scenario.T}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 25, _s.Tillering: 40, _s.Flowering: 70, _s.Senescence: 90})
                        , bbch_application_name=["cabbage 1", "cabbage 2"])
    CI = FOCUSCropMixin(focus_name="Citrus",
                        defined_scenarios=frozenset({Scenario.P, Scenario.O, Scenario.S, Scenario.T}),
                        interception=HashableRSDict({_s.Germination: 70})
                        , bbch_application_name=["cabbage 1", "cabbage 2"])
    CO = FOCUSCropMixin(focus_name="Cotton",
                        defined_scenarios=frozenset({Scenario.S, Scenario.T}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 30, _s.Tillering: 60, _s.Flowering: 75, _s.Senescence: 90})
                        , bbch_application_name=["cotton"])
    GA = FOCUSCropMixin(focus_name="Grass and alfalfa",
                        alternative_names=['Grass', 'Alfalfa'],
                        defined_scenarios=frozenset({Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N,
                                                     Scenario.P, Scenario.O, Scenario.S, Scenario.T}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 40, _s.Tillering: 60, _s.Flowering: 70, _s.Senescence: 90})
                        , bbch_application_name=["Missing"])  # TODO name for bbch lookup
    LS = FOCUSCropMixin(focus_name="Linseed",
                        defined_scenarios=frozenset({Scenario.N}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 30, _s.Tillering: 60, _s.Flowering: 70, _s.Senescence: 90})
                        , bbch_application_name=["linseed"])
    MZ = FOCUSCropMixin(focus_name="Maize",
                        defined_scenarios=frozenset({Scenario.C, Scenario.H, Scenario.K, Scenario.N, Scenario.P,
                                                     Scenario.O, Scenario.S, Scenario.T}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 25, _s.Tillering: 50, _s.Flowering: 75, _s.Senescence: 90}),
                        bbch_application_name=["maize"])
    ON = FOCUSCropMixin(focus_name="Onions",
                        defined_scenarios=frozenset({Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.O,
                                                     Scenario.T}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 10, _s.Tillering: 25, _s.Flowering: 40, _s.Senescence: 60}),
                        bbch_application_name=["onions"])
    OS = FOCUSCropMixin(focus_name="Oilseed rape (summer)",
                        defined_scenarios=frozenset({Scenario.J, Scenario.N, Scenario.O}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 40, _s.Tillering: 80, _s.Flowering: 80, _s.Senescence: 90}),
                        bbch_application_name=["oil seed rape, summer"])
    OW = FOCUSCropMixin(focus_name="Oilseed rape (winter)",
                        defined_scenarios=frozenset({Scenario.C, Scenario.H, Scenario.K, Scenario.N, Scenario.P,
                                                     Scenario.O}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 40, _s.Tillering: 80, _s.Flowering: 80, _s.Senescence: 90}),
                        bbch_application_name=["oil seed rape, winter"])
    PE = FOCUSCropMixin(focus_name="Peas (animals)",
                        defined_scenarios=frozenset({Scenario.C, Scenario.H, Scenario.J, Scenario.N}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 35, _s.Tillering: 55, _s.Flowering: 85, _s.Senescence: 85}),
                        bbch_application_name=["peas"])
    PO = FOCUSCropMixin(focus_name="Potatoes",
                        defined_scenarios=frozenset({Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N,
                                                     Scenario.P, Scenario.O, Scenario.S, Scenario.T}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 15, _s.Tillering: 50, _s.Flowering: 80, _s.Senescence: 50}),
                        bbch_application_name=["potatoes"])
    SB = FOCUSCropMixin(focus_name="Sugar beets",
                        defined_scenarios=frozenset({Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N,
                                                     Scenario.P, Scenario.O, Scenario.S, Scenario.T}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 20, _s.Tillering: 70, _s.Flowering: 90, _s.Senescence: 90}),
                        bbch_application_name=["sugar beet"])
    SC = FOCUSCropMixin(focus_name="Spring cereals",
                        alternative_names=['Cereals, Spring'],
                        defined_scenarios=frozenset({Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N,
                                                     Scenario.O}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 25, _s.Tillering: 50, _s.Elongation: 70, _s.Flowering: 90,
                             _s.Senescence: 90}),
                        bbch_application_name=["cereals spring"])
    SF = FOCUSCropMixin(focus_name="Sunflower",
                        defined_scenarios=frozenset({Scenario.P, Scenario.S}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 20, _s.Tillering: 50, _s.Flowering: 75, _s.Senescence: 90}),
                        bbch_application_name=["sunflower"])
    SO = FOCUSCropMixin(focus_name="Soybeans",
                        defined_scenarios=frozenset({Scenario.P}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 35, _s.Tillering: 55, _s.Flowering: 85, _s.Senescence: 65}),
                        bbch_application_name=["soybean"])
    SW = FOCUSCropMixin(focus_name="Strawberries",
                        defined_scenarios=frozenset({Scenario.H, Scenario.J, Scenario.K, Scenario.S}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 30, _s.Tillering: 50, _s.Flowering: 60, _s.Senescence: 60}),
                        bbch_application_name=["strawberries"])
    TB = FOCUSCropMixin(focus_name="Tobacco",
                        defined_scenarios=frozenset({Scenario.P, Scenario.T}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 50, _s.Tillering: 70, _s.Flowering: 90, _s.Senescence: 90}),
                        bbch_application_name=["tobacco"])
    TM = FOCUSCropMixin(focus_name="Tomatoes",
                        defined_scenarios=frozenset({Scenario.C, Scenario.P, Scenario.O, Scenario.S, Scenario.T}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 50, _s.Tillering: 70, _s.Flowering: 80, _s.Senescence: 50}),
                        bbch_application_name=["tomatoes"])
    VI = FOCUSCropMixin(focus_name="Vines",
                        defined_scenarios=frozenset({Scenario.C, Scenario.H, Scenario.K, Scenario.P, Scenario.O,
                                                     Scenario.S, Scenario.T}),
                        interception=HashableRSDict(
                            {_s.Maturity: 85, _s.DevelopmentFruit: 70, _s.Flowering: 60, _s.Inflorescence: 50,
                             _s.Germination: 40}),
                        bbch_application_name=["vines"])
    WC = FOCUSCropMixin(focus_name="Winter cereals",
                        alternative_names=['Cereals, Winter'],
                        defined_scenarios=frozenset({Scenario.C, Scenario.H, Scenario.J, Scenario.N, Scenario.P,
                                                     Scenario.O}),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 25, _s.Tillering: 50, _s.Elongation: 70, _s.Flowering: 90,
                             _s.Senescence: 90}),
                        bbch_application_name=[
                            "cereals winter"])  # K, S, T have crp files but are not officially defined there

    @staticmethod
    def from_acronym(acronym: str) -> 'FOCUSCrop':
        """Fetches a FOCUSCrop
        :param acronym: the acronym to fetch
        :return: The crop with the name of the acronym
        >>> crop = FOCUSCrop.from_acronym('VI')
        >>> crop.name == 'VI'
        True
        >>> type(crop)
        <enum 'FOCUSCrop'>
        """
        return FOCUSCrop[acronym]

    @staticmethod
    def from_name(name: str) -> 'FOCUSCrop':
        """Fetches a FOCUSCrop by its full name, case-insensitive
        :param name: The full name of the crop
        :return: The FOCUS Crop of that name
        >>> test_crop = FOCUSCrop.from_name("Vines")
        >>> test_crop.name == 'VI'
        True
        >>> FOCUSCrop.from_name("VNes").name
        'VI'"""
        best_crop = FOCUSCrop.AP
        best_ratio = 0
        for crop in FOCUSCrop:
            ratio = SequenceMatcher(None, name.casefold(), crop.focus_name.casefold()).ratio()
            for alt_name in crop.alternative_names:
                alt_ratio = SequenceMatcher(None, name.casefold(), alt_name.casefold()).ratio()
                if alt_ratio > ratio:
                    ratio = alt_ratio
            if ratio > best_ratio:
                best_crop = crop
                best_ratio = ratio
        return best_crop

    @staticmethod
    def parse(parsable: str) -> 'FOCUSCrop':
        """Parse a string into a FOCUSCrop. Use a two letter acronym or a full name, if a full name is used the closest
        match in the registered names is returned"""
        if len(parsable) <= 2:
            return FOCUSCrop.from_acronym(parsable)
        else:
            return FOCUSCrop.from_name(parsable)

    def get_interception(self, bbch: int) -> float:
        """Gets the interception of this plant for a given development stadium.
        Returns no interception for bbch < 0
        :param bbch: The stadium to check
        :return: The interception for that stadium
        >>> FOCUSCrop.VI.get_interception(80)
        85
        >>> FOCUSCrop.VI.get_interception(50)
        50
        >>> FOCUSCrop.VI.get_interception(20)
        40"""
        p = PrincipalStage(min(9, max(0, bbch // 10)))
        for key, value in self.interception.items():
            if p >= key:
                return value
        raise AssertionError("No fitting interception was defined or bbch was not comparable to float")

    def __hash__(self) -> int:
        return hash(tuple(ord(c) for c in self.name))


@dataclass(frozen=True)
class GAP(ABC, TypeCorrecting):
    """An abstract superclass for different ways to create GAPs.
    Most users will want to create GAP Objects with GAP.parse"""
    modelCrop: FOCUSCrop
    """The crop that the field is modelled after"""
    rate: float
    """How much compound will be applied in g/ha"""
    apply_every_n_years: int = 1
    """The time between applications in years. Use this to indicate years without applications"""
    number_of_applications: int = 1
    """How often will be applied"""
    interval: timedelta = 1
    """What is the minimum interval between applications"""
    model_specific_data: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)
    """Any data that only specific models care about will be stored here"""
    name: Optional[str] = field(hash=False, default='')  # str hash is not stable
    """The gaps name. Used only for labelling purposes"""

    @property
    @abstractmethod
    def _type(self) -> str:
        """Which name this class uses during parsing.
        When subclassing, note that this will have to be registered in the GAP.parse method"""
        pass

    @property
    @abstractmethod
    def _dict_args(self) -> Dict[str, Any]:
        """Any special arguments that only this GAP needs, but the others don't"""
        pass

    def _get_common_dict(self) -> Dict[str, Any]:
        """Returns a dict containing the parameters of this GAP that are common to all GAPS"""
        return {
            "name": self.name,
            "modelCrop": self.modelCrop.name if hasattr(self.modelCrop, 'name') else self.modelCrop,
            "rate": self.rate,
            "apply_every_n_years": self.apply_every_n_years,
            "number_of_applications": self.number_of_applications,
            "interval": self.interval.days if type(self.interval) == timedelta else self.interval,
            "model_specific_data": self.model_specific_data
        }

    def __post_init__(self):
        super().__post_init__()
        # To ensure a stable, unique name for a compound that has no name given
        if self.name == '':
            object.__setattr__(self, 'name', f"GAP {hash(self)}")

    @property
    def defined_scenarios(self) -> FrozenSet[Scenario]:
        """Which scenarios this GAP is defined for"""
        return self.modelCrop.defined_scenarios

    @property
    def rate_in_kg(self):
        """A conversion of the application rate into kg"""
        return self.rate / 1000

    def asdict(self) -> Dict:
        """Represents this object as a dictionary"""
        return {
            "type": self._type,
            "arguments": {
                **self._get_common_dict(),
                **self._dict_args
            }
        }

    @abstractmethod
    def application_data(self, scenario: Scenario) -> Generator[Tuple[datetime, float], None, None]:
        """Generates the application specific data for every defined application
        :param scenario: The scenario in which the application happens
        :return: A Generator of tuples in the form (year, month, day, interception)"""
        pass

    @staticmethod
    def parse(to_parse: Dict) -> 'GAP':
        """Parses a dictionary of parameters into a GAP
        :param to_parse: The dictionary containing the GAP definition. Must have 'type' and 'arguments' as keys
        :return: The GAP subclass defined by to_parse"""
        if 'type' not in to_parse.keys():
            raise TypeError("Missing 'type' in to_parse definition")
        if 'arguments' not in to_parse.keys():
            raise TypeError("Missing 'arguments' in to_parse definition")
        types = {
            "relative": RelativeGAP,
            "absolute": AbsoluteConstantGAP,
            "scenario": AbsoluteScenarioGAP,
            "multi": MultiGAP,
        }
        return types[to_parse['type']](**to_parse['arguments'])

    @staticmethod
    def from_excel(excel_file: Path) -> Generator['GAP', None, None]:
        """Parse all GAPS from an Excel file
        :param excel_file: The file to parse
        :return: All gaps defined in the file"""
        gaps = pandas.read_excel(io=excel_file, sheet_name="GAP Properties")
        for _, row in gaps.iterrows():
            yield RelativeGAP(
                name=row['Example GAP'],
                modelCrop=row['Model Crop'],
                rate=row['Rate'],
                number_of_applications=row['Number'],
                interval=row['Interval'],
                bbch=row['BBCH']
            )

    @staticmethod
    def from_path(path: Path) -> Generator['GAP', None, None]:
        """Parse all GAPS in path
        :param path: The path to traverse to find GAPs
        :return: The GAPs found in path"""
        if path.is_dir():
            for file in path.iterdir():
                yield from GAP.from_file(file)
        else:
            yield from GAP.from_file(path)

    @staticmethod
    def from_file(file: Path) -> Generator['GAP', None, None]:
        """Parse a single file for all gaps in it
        :param file: The file to parse
        :return: The GAPs in file"""
        if file.suffix == '.json':
            with file.open() as f:
                json_content = json.load(f)
                if isinstance(json_content, (list, UserList)):
                    yield from (GAP.parse(element) for element in json_content)
                else:
                    yield GAP.parse(json_content)
        elif file.suffix == '.xlsx':
            yield from GAP.from_excel(file)
        elif file.suffix == '.gap':
            yield from GAP.from_gap_machine(file)

    @staticmethod
    def from_gap_machine(file: Path) -> Generator['GAP', None, None]:
        """Create GAPs from the export of the gap machine (go/gap-prod)"""
        yield from GAPMachineGAP.from_gap_machine(file)


@dataclass(frozen=True)
class MultiGAP(GAP):
    """A Container GAP that unifies multiple GAPs that all combine into one.
    Values common to all GAP classes can be specified directly in the MultiGAP definition
    and can then be omitted in the definitions of the individual GAPs or overridden there"""
    timings: Tuple[GAP, ...] = field(default_factory=tuple)
    """The other GAPs that this GAP combines"""

    @property
    def _type(self) -> str:
        return 'multi'

    @property
    def _dict_args(self) -> Dict[str, Any]:
        return {"timings": [{"type": timing._type,
                             "arguments": {key: value
                                           for key, value in timing.asdict()['arguments'].items()
                                           if key not in self._get_common_dict().keys()
                                           or value != self._get_common_dict()[key]}}
                            for timing in self.timings]}

    @property
    def defined_scenarios(self) -> FrozenSet[Scenario]:
        return super().defined_scenarios.intersection(*(timing.defined_scenarios for timing in self.timings))

    def application_data(self, scenario: Scenario) -> Generator[Tuple[datetime, float], None, None]:
        for timing in self.timings:
            yield from timing.application_data(scenario=scenario)

    def __post_init__(self):
        init_dict = self._get_common_dict()
        corrected_timings = tuple()
        for timing in self.timings:
            if isinstance(timing, GAP):
                corrected_timings += tuple([timing])
            else:
                # We are initialising - assume only a dict of values was
                # provided that has to be parsed to a final result
                # noinspection PyTypeChecker
                timing_init: Dict = timing
                init_dict_copy = init_dict.copy()
                init_dict_copy.update(timing_init['arguments'])
                timing_init['arguments'] = init_dict_copy
                corrected_timings += tuple([GAP.parse(timing_init)])
        object.__setattr__(self, 'timings', corrected_timings)
        super().__post_init__()


@dataclass(frozen=True)
class RelativeGAP(GAP):
    """A GAP describing application relative to bbch values.
    It will still yield absolute dates as its application data"""
    bbch: int = 0
    """The BBCH of the first application of the season"""
    season: int = 0
    """Which season this GAP is, 0-indexed, for the crops that have more than one"""

    @property
    def _type(self) -> str:
        return 'relative'

    @property
    def _dict_args(self) -> Dict[str, Any]:
        return {"bbch": self.bbch, "season": self.season}

    def application_data(self, scenario: Scenario) -> Generator[Tuple[datetime, float], None, None]:
        application_line = bbch_to_data_row(self.bbch, scenario,
                                            self.modelCrop.bbch_application_name[self.season])
        time_in_year = application_line['Recommended Application date']
        time_and_interception = tuple()
        for index in range(self.number_of_applications):
            application_time = time_in_year + self.interval * index
            application_line = date_to_data_row(application_time, scenario, self.modelCrop.bbch_application_name[0])
            interception = application_line['Crop Interception(%)']
            time_and_interception += tuple([tuple([application_time, interception])])
        for year in range(1, 6 + 20 * self.apply_every_n_years + 1, self.apply_every_n_years):
            for appl_date, interception in time_and_interception:
                appl_date = datetime(year=year, month=appl_date.month, day=appl_date.day)
                yield appl_date, interception


@dataclass(frozen=True)
class AbsoluteConstantGAP(GAP):
    """A GAP describing an application starting at a given date"""
    time_in_year: datetime = datetime(year=1, month=1, day=1)
    """The date when the first application is, the year component of this value will be ignored. 
    The interval between applications will be added to this date for subsequent applications"""

    @property
    def _type(self) -> str:
        return 'absolute'

    @property
    def _dict_args(self) -> Dict[str, Any]:
        return {"time_in_year": self.time_in_year.isoformat()}

    def application_data(self, scenario: Scenario) -> Generator[Tuple[datetime, float], None, None]:
        time_and_interception: Tuple[Tuple[datetime, float]] = tuple()
        for index in range(self.number_of_applications):
            application_time = self.time_in_year + self.interval * index
            try:
                application_line = date_to_data_row(application_time, scenario, self.modelCrop.bbch_application_name[0])
            except IndexError:
                application_line = date_to_data_row(application_time, scenario, self.modelCrop.bbch_application_name[1])
            interception = application_line['Crop Interception(%)']
            time_and_interception += tuple([tuple([application_time, interception])])
        for year in range(1, 6 + 20 * self.apply_every_n_years + 1, self.apply_every_n_years):
            for appl_date, interception in time_and_interception:
                appl_date = datetime(year=year, month=appl_date.month, day=appl_date.day)
                yield appl_date.replace(year=year), interception

    def __post_init__(self):
        date = None
        if isinstance(self.time_in_year, dict):
            date = datetime(**self.time_in_year)
        elif isinstance(self.time_in_year, (int, float)):
            date = datetime.fromtimestamp(self.time_in_year)
        elif isinstance(self.time_in_year, str):
            date = datetime.fromisoformat(self.time_in_year)
        if date is not None:
            object.__setattr__(self, 'time_in_year', date)
        super().__post_init__()

    @classmethod
    def from_gap(cls, source: GAP, scenario: Scenario) -> 'AbsoluteConstantGAP':
        """Create an AbsoluteConstantGAP from the data of a different GAP"""
        args = source._get_common_dict()
        args['time_in_year'] = next(source.application_data(scenario))[0]
        return cls(**args)


@dataclass(frozen=True)
class AbsoluteDayOfYearGAP(AbsoluteConstantGAP):
    """A GAP to describe an absolute application date by a day of the year instead of date.
    This class is not registered in the GAP.parse logic,
    but can be directly instantiated and then used without issues"""
    day_of_year: int = 0
    """The 0-indexed day of the year for the application date"""

    def __post_init__(self):
        year_start = datetime(2001, 1, 1)
        date_in_2001 = year_start + timedelta(days=self.day_of_year)
        object.__setattr__(self, 'time_in_year', date_in_2001)
        super().__post_init__()


@dataclass(frozen=True)
class AbsoluteScenarioGAP(GAP):
    """A GAP for specifying the day of the year to apply for each GAP individually.
    Behaves similar to the MultiGAP, replacing its List with a Dict[Scenario, GAP] that elides the type determination
    and fixes the type to AbsoluteConstantGAP"""
    scenarios: Dict[Scenario, Dict] = field(
        default_factory=lambda: {}, hash=False)
    """A mapping from scenarios to Constant GAPs"""

    _scenario_gaps: Dict[Scenario, AbsoluteConstantGAP] = field(init=False, repr=False, hash=False, compare=False)

    @property
    def _type(self) -> str:
        return 'scenario'

    @property
    def _dict_args(self) -> Dict[str, Any]:
        return {
            "scenarios": {
                scenario: {
                    key: value
                    for key, value in self._scenario_gaps[scenario].asdict().items()
                    if key not in self._get_common_dict().keys() or self._get_common_dict()[key] != value
                }
                for scenario, gap in self._scenario_gaps.items()
            }
        }

    @property
    def defined_scenarios(self) -> FrozenSet[Scenario]:
        return super().defined_scenarios.intersection(self._scenario_gaps.keys())

    def application_data(self, scenario: Scenario) -> Generator[Tuple[datetime, float], None, None]:
        yield from self._scenario_gaps[scenario].application_data(scenario=scenario)

    def __hash__(self) -> int:
        init_dict = self._get_common_dict()
        init_dict.pop('model_specific_data')
        return hash(tuple([tuple([tuple([key, value]) for key, value in init_dict.items()]),
                           tuple([tuple([key, value]) for key, value in self._scenario_gaps.items()])]))

    def __post_init__(self):
        object.__setattr__(self, 'modelCrop', correct_type(self.modelCrop, FOCUSCrop))
        init_dict = self._get_common_dict()
        corrected_scenarios = {}
        for scenario, scenario_data in self.scenarios.items():
            if isinstance(scenario_data, GAP):
                corrected_scenarios[scenario] = scenario_data
            else:
                # We are initialising - assume only a dict of values was
                # provided that has to be parsed to a final result
                # noinspection PyTypeChecker
                init_dict_copy = init_dict.copy()
                init_dict_copy.update(scenario_data)
                corrected_scenarios[scenario] = AbsoluteConstantGAP(**init_dict_copy)
        object.__setattr__(self, '_scenario_gaps', corrected_scenarios)
        super().__post_init__()


@dataclass(frozen=True)
class GAPMachineGAP(GAP):
    first_season: Dict[Scenario, datetime] = field(default_factory=dict)
    """A mapping from scenario to first application date for the first_season"""
    second_season: Dict[Scenario, datetime] = field(default_factory=dict)
    """A mapping from scenario to first application date for the second_season"""
    interceptions: Tuple[float, ...] = tuple()
    """The interceptions for the application series"""

    def __hash__(self) -> int:
        init_dict = self._get_common_dict()
        init_dict.pop('model_specific_data')
        return hash(tuple([tuple(tuple([key, value]) for key, value in self.first_season.items()),
                           tuple(tuple([key, value]) for key, value in self.second_season.items()),
                           tuple(tuple([key, value]) for key, value in init_dict.items()),
                           self.interceptions]))

    def application_data(self, scenario: Scenario) -> Generator[Tuple[datetime, float], None, None]:
        for year in range(1, 6 + 20 * self.apply_every_n_years + 1, self.apply_every_n_years):
            if scenario in self.first_season.keys() and self.first_season[scenario]:
                for index in range(self.number_of_applications):
                    yield self.first_season[scenario].replace(year=year) + index * self.interval, self.interceptions[
                        index]
            if scenario in self.second_season.keys() and self.second_season[scenario]:
                for index in range(self.number_of_applications):
                    yield self.second_season[scenario].replace(year=year) + index * self.interval, self.interceptions[
                        index]

    @property
    def _dict_args(self) -> Dict[str, Any]:
        return {
            "first_season": self.first_season,
            "second_season": self.second_season,
            "interceptions": self.interceptions
        }

    @property
    def defined_scenarios(self) -> FrozenSet[Scenario]:
        return frozenset(self.first_season.keys()).union(self.second_season.keys())

    @property
    def _type(self) -> str:
        return 'gap_machine'

    @classmethod
    def from_gap_machine(cls, file: Path) -> Generator['GAPMachineGAP', None, None]:
        """Parse the Bayer GAP machine output into GAPs
        :param: The file that is the result from an export in the GAP machine
        :result: The GAPs defined in file"""
        file_object = uncomment(file)
        next(file_object)
        gap_file_reader = csv.DictReader(file_object, delimiter="|",
                                         fieldnames=["PMT ID", "FOCUS Crop used", "FOCUS Crop for interception",
                                                     "GAP Group (DGR) name", "GAP group (DGR) ID", "Use IDs", "Crops",
                                                     "Season", "PMT name", "Repeat-Mode", "Max total # of apps",
                                                     "Appl. mode", "Appl. interval", "User BBCH code", "Appl. Method",
                                                     "Incorporation depth", "1-Châteaudun", "1-Hamburg", "1-Jokioinen",
                                                     "1-Kremsmünster", "1-Okehampton", "1-Piacenza", "1-Porto",
                                                     "1-Sevilla", "1-Thiva", "2-Châteaudun", "2-Hamburg", "2-Jokioinen",
                                                     "2-Kremsmünster", "2-Okehampton", "2-Piacenza", "2-Porto",
                                                     "2-Sevilla", "2-Thiva", "Rate per treatment: ",
                                                     "", "", "", "", "", "", "", ""],
                                         restkey='interceptions')

        for row in gap_file_reader:
            row: Dict[str, Union[str, List[str]]]
            model_crop = FOCUSCrop.parse(row['FOCUS Crop used'])
            gap_name = row['PMT ID']
            rate = float(row['Rate per treatment: '])
            if row['Repeat-Mode'] == "every 3 years":
                period_between_applications = 3
            elif row['Repeat-Mode'] == "every 2 years":
                period_between_applications = 2
            else:
                period_between_applications = 1
            number = int(row['Max total # of apps'])
            interval = timedelta(days=float(row['Appl. interval']))

            # dates are in Lotus123 (and Excel) reckoning, meaning days since 30.12.1899
            first_season = {}
            for scenario in Scenario:
                scenario_name = "1-" + scenario
                if row[scenario_name]:
                    first_season[Scenario(scenario)] = excel_date_to_datetime(int(row[scenario_name]))

            second_season = {}
            for scenario in Scenario:
                scenario_name = "2-" + scenario
                if row[scenario_name]:
                    second_season[Scenario(scenario)] = excel_date_to_datetime(int(row[scenario_name]))
            interceptions = row['interceptions']
            yield cls(modelCrop=model_crop, rate=rate, apply_every_n_years=period_between_applications,
                      number_of_applications=number, interval=interval, model_specific_data={}, name=gap_name,
                      first_season=first_season, second_season=second_season,
                      interceptions=tuple(float(x) for x in interceptions))


# parameters are used in pandas query, which PyCharm does not notice
# noinspection PyUnusedLocal
def bbch_to_data_row(bbch: int, scenario: Scenario, crop_name: str) -> pandas.Series:
    """Given a BBCH, Scenario and crop, find the first valid entry in the application table
    :param bbch: The BBCH for the lookup. Needs to be in the range [0,99]
    :param scenario: The scenario for the lookup. Must be defined for the given crop
    :param crop_name: The name of the crop
    :return: A pandas series containing the following values:
    Location,Crop,Requested BBCH Code,Allocated BBCH Code,Recommended Application date,Crop Interception(%)"""
    scenario_name = scenario.replace("â", "a").replace("ü", "u")
    return bbch_application.query('Location == @scenario_name '
                                  '& Crop == @crop_name'
                                  '& `Requested BBCH Code` == @bbch').iloc[0]


# parameters are used in pandas query, which PyCharm does not notice
# noinspection PyUnusedLocal
def date_to_data_row(date: datetime, scenario: Scenario, crop_name: str) -> pandas.Series:
    """Given a date, Scenario and crop, find the first valid entry in the application table
    :param date: The date for the lookup. Needs to be in the winter of 2001 or the summer of 2002
    :param scenario: The scenario for the lookup. Must be defined for the given crop
    :param crop_name: The name of the crop
    :return: A pandas series containing the following values:
    Location,Crop,Requested BBCH Code,Allocated BBCH Code,Recommended Application date,Crop Interception(%)"""
    scenario_name = scenario.replace("â", "a").replace("ü", "u")
    filtered_frame: pandas.DataFrame = bbch_application.query('Location == @scenario_name '
                                                              '& Crop == @crop_name'
                                                              '& `Recommended Application date` >= '
                                                              '@date'
                                                              )
    return filtered_frame.sort_values(by=['Recommended Application date']).iloc[0]
