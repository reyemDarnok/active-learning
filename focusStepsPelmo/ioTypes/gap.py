import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Generator, List, Tuple, NamedTuple, Any, OrderedDict, Optional

import numpy
import pandas

from focusStepsPelmo.util.conversions import excel_date_to_datetime
from focusStepsPelmo.util.datastructures import HashableRSDict, TypeCorrecting

bbch_application: pandas.DataFrame = pandas.read_csv(Path(__file__).parent / 'BBCHGW.csv',
                                                     header=0,
                                                     dtype={'Location': 'category', 'Crop': 'category',
                                                            'Requested BBCH Code': 'byte',
                                                            'Allocated BBCH Code': 'byte',
                                                            'Crop Interception(%)': 'byte'},
                                                     parse_dates=['Recommended Application date'],
                                                     dayfirst=True)
bbch_application['Location'] = bbch_application['Location'].cat.rename_categories({'Kremsmunster': 'Kremsmünster',
                                                                                   'Chateaudun': 'Châteaudun'})


class Scenario(str, Enum):
    """The Pelmo Scenarios. The key is the one letter shorthand and the value the full name"""
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
    defined_scenarios: Tuple[Scenario, ...]
    """The scenarios that are defined for this Crop in Pelmo"""
    interception: OrderedDict[PrincipalStage, float]
    """Mapping bbch states to interception values"""
    bbch_application_name: List[str]
    """The category in the bbch_application data this crop belongs to"""


_s = PrincipalStage


# noinspection PyTypeChecker
class FOCUSCrop(FOCUSCropMixin, Enum):
    """The crops defined for Pelmo. Each defined as a PelmoCropMixin"""
    AP = FOCUSCropMixin(focus_name="Apples",
                        defined_scenarios=tuple(
                            [Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O,
                             Scenario.S, Scenario.T]),
                        interception=HashableRSDict(
                            {_s.Maturity: 80, _s.DevelopmentFruit: 70, _s.Flowering: 65, _s.Germination: 50})
                        , bbch_application_name=["pome fruit"])
    BB = FOCUSCropMixin(focus_name="Bush berries",
                        defined_scenarios=tuple([Scenario.J]),
                        interception=HashableRSDict(
                            {_s.Maturity: 80, _s.DevelopmentFruit: 65, _s.Flowering: 65, _s.Germination: 50})
                        , bbch_application_name=["Missing"])  # TODO)
    BF = FOCUSCropMixin(focus_name="Beans (field)",
                        defined_scenarios=tuple([Scenario.H, Scenario.K, Scenario.N]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 25, _s.Tillering: 40, _s.Flowering: 70, _s.Senescence: 80})
                        , bbch_application_name=["beans 1", "beans 2"])
    BV = FOCUSCropMixin(focus_name="Beans (vegetables)",
                        defined_scenarios=tuple([Scenario.O, Scenario.T]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 25, _s.Tillering: 40, _s.Flowering: 70, _s.Senescence: 80})
                        , bbch_application_name=["beans 1", "beans 2"])
    CA = FOCUSCropMixin(focus_name="Carrots",
                        defined_scenarios=tuple(
                            [Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.O, Scenario.T]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 25, _s.Tillering: 60, _s.Flowering: 80, _s.Senescence: 80})
                        , bbch_application_name=["carrots 1", "carrots 2"])
    CB = FOCUSCropMixin(focus_name="Cabbage",
                        defined_scenarios=tuple(
                            [Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.O, Scenario.S, Scenario.T]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 25, _s.Tillering: 40, _s.Flowering: 70, _s.Senescence: 90})
                        , bbch_application_name=["cabbage 1", "cabbage 2"])
    CI = FOCUSCropMixin(focus_name="Citrus",
                        defined_scenarios=tuple([Scenario.P, Scenario.O, Scenario.S, Scenario.T]),
                        interception=HashableRSDict({_s.Germination: 70})
                        , bbch_application_name=["cabbage 1", "cabbage 2"])
    CO = FOCUSCropMixin(focus_name="Cotton",
                        defined_scenarios=tuple([Scenario.S, Scenario.T]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 30, _s.Tillering: 60, _s.Flowering: 75, _s.Senescence: 90})
                        , bbch_application_name=["cotton"])
    GA = FOCUSCropMixin(focus_name="Grass and alfalfa",
                        defined_scenarios=tuple(
                            [Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O,
                             Scenario.S, Scenario.T]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 40, _s.Tillering: 60, _s.Flowering: 70, _s.Senescence: 90})
                        , bbch_application_name=["Missing"])  # TODO)
    LS = FOCUSCropMixin(focus_name="Linseed",
                        defined_scenarios=tuple([Scenario.N]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 30, _s.Tillering: 60, _s.Flowering: 70, _s.Senescence: 90})
                        , bbch_application_name=["linseed"])
    MZ = FOCUSCropMixin(focus_name="Maize",
                        defined_scenarios=tuple(
                            [Scenario.C, Scenario.H, Scenario.K, Scenario.N, Scenario.P, Scenario.O, Scenario.S,
                             Scenario.T]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 25, _s.Tillering: 50, _s.Flowering: 75, _s.Senescence: 90}),
                        bbch_application_name=["maize"])
    ON = FOCUSCropMixin(focus_name="Onions",
                        defined_scenarios=tuple(
                            [Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.O, Scenario.T]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 10, _s.Tillering: 25, _s.Flowering: 40, _s.Senescence: 60}),
                        bbch_application_name=["onions"])
    OS = FOCUSCropMixin(focus_name="Oilseed rape (summer)",
                        defined_scenarios=tuple([Scenario.J, Scenario.N, Scenario.O]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 40, _s.Tillering: 80, _s.Flowering: 80, _s.Senescence: 90}),
                        bbch_application_name=["oil seed rape, summer"])
    OW = FOCUSCropMixin(focus_name="Oilseed rape (winter)",
                        defined_scenarios=tuple(
                            [Scenario.C, Scenario.H, Scenario.K, Scenario.N, Scenario.P, Scenario.O]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 40, _s.Tillering: 80, _s.Flowering: 80, _s.Senescence: 90}),
                        bbch_application_name=["oil seed rape, winter"])
    PE = FOCUSCropMixin(focus_name="Peas (animals)",
                        defined_scenarios=tuple([Scenario.C, Scenario.H, Scenario.J, Scenario.N]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 35, _s.Tillering: 55, _s.Flowering: 85, _s.Senescence: 85}),
                        bbch_application_name=["peas"])
    PO = FOCUSCropMixin(focus_name="Potatoes",
                        defined_scenarios=tuple(
                            [Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O,
                             Scenario.S, Scenario.T]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 15, _s.Tillering: 50, _s.Flowering: 80, _s.Senescence: 50}),
                        bbch_application_name=["potatoes"])
    SB = FOCUSCropMixin(focus_name="Sugar beets",
                        defined_scenarios=tuple(
                            [Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.P, Scenario.O,
                             Scenario.S, Scenario.T]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 20, _s.Tillering: 70, _s.Flowering: 90, _s.Senescence: 90}),
                        bbch_application_name=["sugar beet"])
    SC = FOCUSCropMixin(focus_name="Spring cereals",
                        defined_scenarios=tuple(
                            [Scenario.C, Scenario.H, Scenario.J, Scenario.K, Scenario.N, Scenario.O]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 25, _s.Tillering: 50, _s.Elongation: 70, _s.Flowering: 90,
                             _s.Senescence: 90}),
                        bbch_application_name=["cereals spring"])
    SF = FOCUSCropMixin(focus_name="Sunflower",
                        defined_scenarios=tuple([Scenario.P, Scenario.S]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 20, _s.Tillering: 50, _s.Flowering: 75, _s.Senescence: 90}),
                        bbch_application_name=["sunflower"])
    SO = FOCUSCropMixin(focus_name="Soybeans",
                        defined_scenarios=tuple([Scenario.P]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 35, _s.Tillering: 55, _s.Flowering: 85, _s.Senescence: 65}),
                        bbch_application_name=["soybean"])
    SW = FOCUSCropMixin(focus_name="Strawberries",
                        defined_scenarios=tuple([Scenario.H, Scenario.J, Scenario.K, Scenario.S]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 30, _s.Tillering: 50, _s.Flowering: 60, _s.Senescence: 60}),
                        bbch_application_name=["strawberries"])
    TB = FOCUSCropMixin(focus_name="Tobacco",
                        defined_scenarios=tuple([Scenario.P, Scenario.T]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 50, _s.Tillering: 70, _s.Flowering: 90, _s.Senescence: 90}),
                        bbch_application_name=["tobacco"])
    TM = FOCUSCropMixin(focus_name="Tomatoes",
                        defined_scenarios=tuple([Scenario.C, Scenario.P, Scenario.O, Scenario.S, Scenario.T]),
                        interception=HashableRSDict(
                            {_s.Germination: 0, _s.Leaf: 50, _s.Tillering: 70, _s.Flowering: 80, _s.Senescence: 50}),
                        bbch_application_name=["tomatoes"])
    VI = FOCUSCropMixin(focus_name="Vines",
                        defined_scenarios=tuple(
                            [Scenario.C, Scenario.H, Scenario.K, Scenario.P, Scenario.O, Scenario.S, Scenario.T]),
                        interception=HashableRSDict(
                            {_s.Maturity: 85, _s.DevelopmentFruit: 70, _s.Flowering: 60, _s.Inflorescence: 50,
                             _s.Germination: 40}),
                        bbch_application_name=["vines"])
    WC = FOCUSCropMixin(focus_name="Winter cereals",
                        defined_scenarios=tuple(
                            [Scenario.C, Scenario.H, Scenario.J, Scenario.N, Scenario.P, Scenario.O]),
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
    def from_name(name: str) -> Optional['FOCUSCrop']:
        """Fetches a FOCUSCrop by its full name, case-insensitive
        :param name: The full name of the crop
        :return: The FOCUS Crop of that name
        >>> test_crop = FOCUSCrop.from_name("Vines")
        >>> test_crop.name == 'VI'
        True"""
        for crop in FOCUSCrop:
            if crop.focus_name.casefold() == name.casefold():
                return crop
        return None

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
    modelCrop: FOCUSCrop
    """The crop that the field is modelled after"""
    rate: float
    """How much compound will be applied in g/ha"""
    period_between_applications: int = 1
    number: int = 1
    """How often will be applied"""
    interval: timedelta = 1
    """What is the minimum interval between applications"""
    model_specific_data: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @abstractmethod
    def asdict(self) -> Dict:
        return {'modelCrop': self.modelCrop.name,
                'rate': self.rate,
                "period_between_applications": self.period_between_applications,
                'number': self.number,
                'interval': self.interval}

    @abstractmethod
    def application_data(self, scenario: Scenario) -> Generator[Tuple[datetime, float], None, None]:
        """Generates the application specific data for every defined application
        :param scenario: The scenario in which the application happens
        :return: A Generator of tuples in the form (year, month, day, interception)"""
        pass

    @staticmethod
    def parse(to_parse: Dict) -> 'GAP':
        types = {
            "relative": RelativeGAP,
            "absolute": AbsoluteConstantGAP,
            "scenario": AbsoluteScenarioGAP,
            "multi": MultiGAP,
        }
        return types[to_parse['type']](**to_parse['arguments'])

    @staticmethod
    def from_excel(excel_file: Path) -> List['GAP']:
        gaps = pandas.read_excel(io=excel_file, sheet_name="GAP Properties")
        return [RelativeGAP(
            modelCrop=row['Model Crop'],
            rate=row['Rate'],
            number=row['Number'],
            interval=row['Interval'],
            bbch=row['BBCH']
        )
            for _, row in gaps.iterrows()]

    @staticmethod
    def from_path(path: Path) -> Generator['GAP', None, None]:
        if path.is_dir():
            for file in path.iterdir():
                yield from GAP.from_file(file)
        else:
            yield from GAP.from_file(path)

    @staticmethod
    def from_file(file: Path) -> Generator['GAP', None, None]:
        if file.suffix == '.json':
            with file.open() as f:
                json_content = json.load(f)
                try:
                    yield from (GAP.parse(element) for element in json_content)
                except TypeError:
                    yield GAP.parse(json_content)
        elif file.suffix == '.xlsx':
            yield from GAP.from_excel(file)
        elif file.suffix == '.gap':
            yield from GAP.from_gap_machine(file)

    @classmethod
    def from_gap_machine(cls, file: Path) -> Generator['GAP', None, None]:
        with file.open() as gap_file:
            gap_file.readline()
            gap_file.readline()
            gap_file.readline()
            csv_reader = csv.reader(gap_file, delimiter='|')
            header = next(csv_reader)
            header.append('Interceptions.1')
            seen = set()
            filtered_header = []
            for h in header:
                if h:
                    if h not in seen:
                        filtered_header.append(h)
                        seen.add(h)
                    else:
                        candidate_num = 1
                        while f"{h}.{candidate_num}" in seen:
                            candidate_num += 1
                        filtered_header.append(f"{h}.{candidate_num}")
                        seen.add(f"{h}.{candidate_num}")
        gap_df = pandas.read_csv(file, skiprows=4, sep='|', skip_blank_lines=True, names=filtered_header,
                                 memory_map=True, usecols=[*range(35), 43, 44])
        for _, row in gap_df.iterrows():
            model_crop = FOCUSCrop.from_name(row['FOCUS Crop used'])
            model_data = {'gap_machine': {'PMT ID': row['PMT ID']}}
            rate = row['Rate per treatment: ']
            if row['Repeat-Mode'] == "every 3 years":
                period_between_applications = 3
            elif row['Repeat-Mode'] == "every 2 years":
                period_between_applications = 2
            else:
                period_between_applications = 1
            number = row['Max total # of apps']
            interval = row['Appl. intervall']  # typo in source data, intentional here

            # dates are in Lotus123 (and Excel) reckoning, meaning days since 30.12.1899
            scenarios = {}
            if not numpy.isnan(row["Appl. dates, 1st veg. period Chateaudun"]):
                scenarios[Scenario.C] = excel_date_to_datetime(row['Appl. dates, 1st veg. period Chateaudun'])
            for scenario in Scenario:
                if scenario == Scenario.C:
                    continue
                scenario_name = scenario.replace('ü', 'u')
                if not numpy.isnan(row[scenario_name]):
                    scenarios[Scenario(scenario)] = excel_date_to_datetime(row[scenario_name])
            first_gap = AbsoluteScenarioGAP(modelCrop=model_crop, model_specific_data=model_data, rate=rate,
                                            number=number, period_between_applications=period_between_applications,
                                            interval=interval,
                                            scenarios=scenarios)
            second_scenarios = {}
            if not numpy.isnan(row["Appl. dates, 2nd veg. period Chateaudun"]):
                scenarios[Scenario.C] = excel_date_to_datetime(row['Appl. dates, 2nd veg. period Chateaudun'])
            for scenario in Scenario:
                if scenario == Scenario.C:
                    continue
                scenario_name = scenario.replace('ü', 'u') + ".1"
                if not numpy.isnan(row[scenario_name]):
                    scenarios[Scenario(scenario)] = excel_date_to_datetime(row[scenario_name])
            if second_scenarios:
                second_gap = AbsoluteScenarioGAP(modelCrop=model_crop, model_specific_data=model_data, rate=rate,
                                                 number=number, period_between_applications=period_between_applications,
                                                 interval=interval,
                                                 scenarios=second_scenarios)
                MultiGAP(modelCrop=model_crop, model_specific_data=model_data, rate=rate,
                         number=number, period_between_applications=period_between_applications,
                         interval=interval,
                         timings=(first_gap, second_gap))
            else:
                yield first_gap


# parameters are used in pandas query, which PyCharm does not notice
# noinspection PyUnusedLocal
def bbch_to_data_row(bbch: int, scenario: Scenario, crop_name: str) -> pandas.Series:
    return bbch_application.query('Location == @scenario.name '
                                  '& Crop == @crop_name'
                                  '& `Requested BBCH Code` == @bbch').iloc[0]


# parameters are used in pandas query, which PyCharm does not notice
# noinspection PyUnusedLocal
def date_to_data_row(date: datetime, scenario: Scenario, crop_name: str) -> pandas.Series:
    filtered_frame: pandas.DataFrame = bbch_application.query('Location == @scenario.value '
                                                              '& Crop == @crop_name'
                                                              '& `Recommended Application date` > '
                                                              '@date'
                                                              )
    return filtered_frame.sort_values(ascending=True, by=['Recommended Application date']).iloc[0]


@dataclass(frozen=True)
class MultiGAP(GAP):
    def asdict(self) -> Dict:
        super_dict = super().asdict()
        super_dict.update({'timings': self.timings})
        return super_dict

    def application_data(self, scenario: Scenario) -> Generator[Tuple[datetime, float], None, None]:
        for timing in self.timings:
            yield from timing.application_data(scenario)

    def __post_init__(self):
        init_dict = self.asdict()
        init_dict.pop('timings')
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
                init_dict_copy.update(timing_init)
                corrected_timings += tuple([GAP.parse(init_dict_copy)])
        object.__setattr__(self, 'timings', corrected_timings)
        super().__post_init__()

    timings: Tuple[GAP, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RelativeGAP(GAP):
    def application_data(self, scenario: Scenario) -> Generator[Tuple[datetime, float], None, None]:
        application_line = bbch_to_data_row(self.bbch, scenario,
                                            self.modelCrop.bbch_application_name[self.season])
        time_in_year = application_line['Recommended Application date']
        time_and_interception = tuple()
        for index in range(self.number):
            application_time = time_in_year + self.interval * index
            application_line = date_to_data_row(application_time, scenario, self.modelCrop.bbch_application_name[0])
            interception = application_line['Crop Interception(%)']
            time_and_interception += application_time, interception
        for year in range(1, 6 + 20 * self.period_between_applications + 1, self.period_between_applications):
            for appl_date, interception in time_and_interception:
                appl_date = appl_date.replace(year=year)
                yield appl_date, interception

    def asdict(self) -> Dict:
        super_dict = super().asdict()
        super_dict.update({'bbch': self.bbch})
        return super_dict

    bbch: int = 0
    season: int = 0


@dataclass(frozen=True)
class AbsoluteConstantGAP(GAP):
    time_in_year: datetime = datetime(year=1, month=1, day=1)

    def application_data(self, scenario: Scenario) -> Generator[Tuple[datetime, float], None, None]:
        time_and_interception: Tuple[Tuple[datetime, float]] = tuple()
        for index in range(self.number):
            application_time = self.time_in_year + self.interval * index
            try:
                application_line = date_to_data_row(application_time, scenario, self.modelCrop.bbch_application_name[0])
            except IndexError:
                application_line = date_to_data_row(application_time, scenario, self.modelCrop.bbch_application_name[1])
            interception = application_line['Crop Interception(%)']
            time_and_interception += tuple([tuple([application_time, interception])])
        for year in range(1, 6 + 20 * self.period_between_applications + 1, self.period_between_applications):
            for date, interception in time_and_interception:
                yield date.replace(year=year), interception

    def asdict(self) -> Dict:
        super_dict = super().asdict()
        super_dict.update({'time_in_year': self.time_in_year})
        return super_dict

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


@dataclass(frozen=True)
class AbsoluteDayOfYearGAP(AbsoluteConstantGAP):
    day_of_year: int = 0

    def __post_init__(self):
        year_start = datetime(2001, 1, 1)
        date_in_2001 = year_start + timedelta(days=self.day_of_year)
        object.__setattr__(self, 'time_in_year', date_in_2001)
        super().__post_init__()


@dataclass(frozen=True)
class AbsoluteScenarioGAP(GAP):
    def application_data(self, scenario: Scenario) -> Generator[Tuple[datetime, float], None, None]:
        yield from self._scenario_gaps[scenario].application_data(scenario)

    scenarios: Dict[Scenario, datetime] = field(
        default_factory=lambda: {})

    _scenario_gaps: Dict[Scenario, AbsoluteConstantGAP] = field(init=False, repr=False, hash=False, compare=False)

    def __post_init__(self):
        init_dict = self.asdict()
        init_dict.pop('scenarios')
        scenario_gaps = {scenario: AbsoluteConstantGAP(time_in_year=date, **init_dict)
                         for scenario, date in self.scenarios.items()}
        object.__setattr__(self, '_scenario_gaps', scenario_gaps)
        super().__post_init__()

    def asdict(self) -> Dict:
        super_dict = super().asdict()
        super_dict.update({'scenarios': self.scenarios})
        return super_dict
