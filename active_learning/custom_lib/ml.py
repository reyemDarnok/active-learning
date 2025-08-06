from enum import Enum
import json
import math
import tempfile
import textwrap
import numpy
import numpy.typing as npt
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Generator, Iterable, List, NoReturn, Optional, Sequence, Tuple
from datetime import timedelta
from pathlib import Path

import pandas
from sys import path

from sklearn.base import BaseEstimator
from sklearn.preprocessing import  StandardScaler
from focusStepsPelmo.ioTypes.scenario import Scenario
from focusStepsPelmo.pelmo.local import run_local
from focusStepsPelmo.util.conversions import EnhancedJSONEncoder, flatten, flatten_to_keys
path.append(str(Path(__file__).parent.parent.parent))
from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.pelmo.generation_definition import Definition
from modAL.models.base import BaseCommittee
from modAL.utils.selection import multi_argmax
 

class PartialScaler:
    def __init__(self, to_exclude: Sequence[str], **scaler_kwargs):
        self.scaler = StandardScaler(**scaler_kwargs)
        self.to_exclude = to_exclude

    def fit(self, X: pandas.DataFrame, *args, **kwargs):
        self.scaler.fit(X.drop(columns=self.to_exclude), *args, **kwargs)
        return self


    def transform(self, X: pandas.DataFrame, *args, **kwargs):
        passthrough_columns = {name: X[name] for name in self.to_exclude}
        transforming_columns = [name for name in X.columns if name not in self.to_exclude]
        res = pandas.DataFrame(self.scaler.transform(X.drop(columns=self.to_exclude), *args, **kwargs), columns=transforming_columns)
        for name, column in passthrough_columns.items():
            res[name] =[int(x) for x in column]
        return res

def split_into_data_and_label(dataset: pandas.DataFrame) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    #pecs = dataset.columns[dataset.columns.str.endswith('.pec')]
    pecs = ["parent.pec"]
    data = dataset.drop(pecs, axis=1)
    label = dataset[pecs].copy()
    for column in label:
        label[column] = label[column].apply(lambda x: math.log10(x))
    return data, label[pecs]

def split_into_data_and_label_raw(dataset: pandas.DataFrame) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    #pecs = dataset.columns[dataset.columns.str.endswith('.pec')]
    pecs = ["0.compound_pec"]
    data = dataset.drop(pecs, axis=1)
    data.drop(columns=['0.compound_name'], inplace=True)
    label = dataset[pecs].copy()
    for column in label:
        label[column] = label[column].apply(lambda x: math.log10(x))
    return data, label[pecs]

def GP_regression_std(regressor, X, n_instances=1):
    _, std = regressor.predict(X, return_std=True)
    idx = numpy.argpartition(std, -n_instances)[-n_instances:]
    return idx, X[idx]

def random_sample(_, X:pandas.DataFrame, n_instances:int=1):
    return X.sample(n=n_instances, replace=True).index

def generate_features(template: Definition, number: int):
    combination_gen = _combination_generator(template=template)
    combinations = numpy.fromiter(combination_gen, object, count=number)
    flattened = [list(flatten({'combination': combination.asdict()})) for combination in combinations]
    return combinations, pandas.DataFrame(flattened, columns=list(flatten_to_keys({'combination': combinations[0].asdict()})))

def _combination_generator(template: Definition) -> Generator[Combination, Any, NoReturn]:
    while True:
        try:
            candidate = Combination(**template.make_sample())
            list(candidate.gap.application_data(next(candidate.scenarios.__iter__())))
            yield candidate
        except (ValueError, IndexError):
            pass

def evaluate_features(features: Iterable[Combination]) -> pandas.DataFrame:
    feature_tuple = tuple(features)
    name = hash(feature_tuple)
    try:
        with tempfile.TemporaryDirectory() as work:
            work_dir = Path(work)
            combination_path = (work_dir / 'combination' / f"{name}.json")
            combination_path.parent.mkdir(exist_ok=True, parents=True)
            with combination_path.open('w') as combination_file:
                json.dump(feature_tuple, fp=combination_file, cls=EnhancedJSONEncoder)
            pelmo_res_path = work_dir / 'pelmo_result' / f"{name}.csv"
            run_local(output_file=pelmo_res_path,combination_dir=combination_path)
            result_df = pandas.read_csv(pelmo_res_path)
            return result_df
    except FileNotFoundError:
        # something went wrong while calculating - I don't know why the context manager files and not something else but thats the error
        with open('erroring.json', w) errorfile:
            json.dump([x.asdict() for x in feature_tuple], errorfile)
        raise

       
    
@dataclass
class Score:
    value: float = numpy.NAN
    minimum: float = numpy.NAN
    maximum: float = numpy.NAN
    std: float = numpy.NAN
    
    def to_json(self):
        return asdict(self)
@dataclass
class ScenarioScores:
    metric: Callable
    combined: List['Score'] = field(default_factory=list)
    scenarios: Dict[Scenario, List['Score']] = field(default_factory=lambda: {s: [] for s in Scenario})
    
    def __repr__(self):
        return f"ScenarioScores(combined={self.combined[-1] if self.combined else {}},scenarios={{key.name: value[-1] for key, value in self.scenarios.items() if value and not numpy.isnan(value[-1].value)}})"

    def __str__(self):
       return str(self.combined[-1] if self.combined else {})
    
    def to_json(self):
        return {"combined": self.combined[-1].to_json() if self.combined else {},
                "scenarios": {key.name: value[-1].to_json() for key, value in self.scenarios.items() if value and not numpy.isnan(value[-1].value)}
                }

    def update_scores(self, learner: BaseCommittee, test_labels: pandas.DataFrame, test_features: pandas.DataFrame) -> None:
        if test_features.shape[0] > 0:
            individual_predict_labels = [indiv.predict(test_features) for indiv in learner.learner_list]
            committee_predict_labels = learner.predict(test_features)
        else:
            individual_predict_labels = [numpy.ndarray((0,1)) for _ in learner.learner_list]
            committee_predict_labels = numpy.ndarray((0,1))
        for index, scenario in enumerate(Scenario):
            scenario_index = (test_features['combination.scenarios.0'] == scenario.name) | (test_features['combination.scenarios.0'] == index) 
            self.scenarios[scenario].append(
                make_score(metric=self.metric, 
                        test_labels=test_labels[scenario_index], 
                        individual_predict_labels=[pred[scenario_index] for pred in individual_predict_labels],
                        committee_predict_labels=committee_predict_labels[scenario_index])
            )
        self.combined.append(
            make_score(metric=self.metric, 
                    test_labels=test_labels, 
                    individual_predict_labels=individual_predict_labels, 
                    committee_predict_labels=committee_predict_labels)
                    )

@dataclass
class DatasetScores:
    total_features: pandas.DataFrame = field(repr=False)
    total_labels: pandas.DataFrame = field(repr=False)
    dataset_filters: Dict[str, Callable[[pandas.DataFrame, pandas.DataFrame], 'pandas.Series[bool]']] = field(repr=False, default_factory=dict)
    metric: Callable[[npt.ArrayLike, npt.ArrayLike], float] = field(repr=False, default=lambda x,y: 0)
    scores: Dict[str, ScenarioScores] = field(default_factory=dict)
    _filtered_features: Dict[str, pandas.DataFrame] = field(default_factory=dict, repr=False)
    _filtered_labels: Dict[str, pandas.DataFrame] = field(default_factory=dict, repr=False)

    
    def to_json(self):
        return {
            key: value.to_json() for key, value in self.scores.items()
        }

    def __post_init__(self):
        assert self.total_features.shape[0] == self.total_labels.shape[0]
        self._filter()
        for name in self.dataset_filters.keys():
            self.scores[name] = ScenarioScores(metric=self.metric)


    def _filter(self):
        for name, filter in self.dataset_filters.items():
            index = filter(self.total_features, self.total_labels)
            self._filtered_features[name] = self.total_features[index]
            self._filtered_labels[name] = self.total_labels[index]

    def update_scores(self, learner: BaseCommittee, total_features: Optional[pandas.DataFrame] = None, total_labels: Optional[pandas.DataFrame] = None) -> None:
        assert (total_features is None) == (total_labels is None)
        if total_features is not None:
            assert total_labels is not None
            assert total_features.shape[0] == total_labels.shape[0]
            self.total_features = total_features
            self.total_labels = total_labels
            self._filter()

        for name, scores in self.scores.items():
            scores.update_scores(learner=learner, test_labels=self._filtered_labels[name], test_features=self._filtered_features[name])

class Category(str, Enum):
    TRAIN = "Internal Test"
    CONFIRM = "External Test"

@dataclass
class CategoryScores:
    dataset_scores: Dict[str, DatasetScores] = field(default_factory=dict)

    def to_json(self):
        return {
            key: value.to_json() for key, value in self.dataset_scores.items()
        }

@dataclass
class TrainingRecord:
    model: BaseCommittee = field(repr=False)
    batchsize: int = 0
    scores: Dict[Category, CategoryScores] = field(default_factory=lambda: {category: CategoryScores() for category in Category})
    training_times: List[timedelta] = field(default_factory=list)
    training_sizes: List[int] = field(default_factory=list)
    all_training_points: pandas.DataFrame = field(default=None, repr=False) # type: ignore
    usable_points: int = 0
    attempted_points: int = 0

    def to_json(self):
        
        return {"batchsize": self.batchsize, 
                "scores": {key: value.to_json() for key, value in self.scores.items()},
                "training_size": self.training_sizes[-1]
                }
 

def make_score(metric, test_labels, individual_predict_labels, committee_predict_labels) -> Score:
    if committee_predict_labels.shape[0] > 0:
        try:
            score = metric(individual_predict_labels)
            return Score(score, score, score)
        except:
            committee_score = metric(test_labels, committee_predict_labels)
            individual_scores = [metric(test_labels, indiv_predict_labels) 
                                for indiv_predict_labels in individual_predict_labels]
            return Score(value=committee_score, minimum=min(individual_scores), maximum=max(individual_scores), std=numpy.std(individual_scores)) # type: ignore
    else:
        return Score()

def skewed_std_sampling(regressor: BaseCommittee, X: pandas.DataFrame, n_instances: int = 1):
    skew_target = -1
    max_weight = 10
    min_weight = 1
    falloff = 2
    predicts, std = regressor.predict(X, return_std=True) # type: ignore
    weights = numpy.maximum(min_weight, max_weight - numpy.abs(predicts - skew_target) * falloff) * std
    return multi_argmax(weights, n_instances=n_instances)
