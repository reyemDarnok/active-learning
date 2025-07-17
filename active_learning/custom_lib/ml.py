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

from focusStepsPelmo.ioTypes.scenario import Scenario
from focusStepsPelmo.pelmo.local import run_local
from focusStepsPelmo.util.conversions import EnhancedJSONEncoder, flatten, flatten_to_keys
path.append(str(Path(__file__).parent.parent.parent))
from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.pelmo.generation_definition import Definition
from modAL.models.base import BaseCommittee
from modAL.utils.selection import multi_argmax


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
            yield Combination(**template.make_sample())
        except ValueError:
            pass

def evaluate_features(features: Iterable[Combination]) -> pandas.DataFrame:
    feature_tuple = tuple(features)
    name = hash(feature_tuple)
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

       
    
@dataclass
class Score:
    value: float = 0
    minimum: float = 0
    maximum: float = 0

    def __str__(self):
        return f"value={self.value:.2f}\nminimum={self.minimum:.2f}\nmaximum={self.maximum:.2f}\n"
    
    def to_json(self):
        return asdict(self)
@dataclass
class ScenarioScores:
    metric: Callable
    combined: List['Score'] = field(default_factory=list)
    scenarios: Dict[Scenario, List['Score']] = field(default_factory=lambda: {s: [] for s in Scenario})
    
    def __str__(self):
       return str(self.combined[-1] if self.combined else numpy.NAN)
    
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
        for scenario in Scenario:
            scenario_index = test_features['combination.scenarios.0'] == scenario.name
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
    total_features: pandas.DataFrame
    total_labels: pandas.DataFrame
    dataset_filters: Dict[str, Callable[[pandas.DataFrame, pandas.DataFrame], 'pandas.Series[bool]']] = field(repr=False, default_factory=dict)
    metric: Callable[[npt.ArrayLike, npt.ArrayLike], float] = field(repr=False, default=lambda x,y: 0)
    scores: Dict[str, ScenarioScores] = field(default_factory=dict)
    _filtered_features: Dict[str, pandas.DataFrame] = field(default_factory=dict)
    _filtered_labels: Dict[str, pandas.DataFrame] = field(default_factory=dict)

    def __str__(self):
        res = ""
        for key, value in self.scores.items():
            res += f"{key}=\n{textwrap.indent(str(value), chr(9))}"
        textwrap.indent(res, ' ')
        return res
    
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

    
@dataclass
class TrainingRecord:
    model: BaseCommittee = field(repr=False)
    batchsize: int = 0
    validation_scores: Dict[str, 'DatasetScores'] = field(default_factory=dict)
    test_scores: Dict[str, 'DatasetScores'] = field(default_factory=dict)
    training_times: List[timedelta] = field(default_factory=list)
    training_sizes: List[int] = field(default_factory=list)
    all_training_points: pandas.DataFrame = None # type: ignore
    usable_points: int = 0
    attempted_points: int = 0

    def __str__(self):
        validation_scores = ""
        for key, value in self.validation_scores.items():
            validation_scores += f"{key}=\n{textwrap.indent(str(value), chr(9))}"
        validation_scores = textwrap.indent(validation_scores, '\t')
        test_scores = ""
        for key, value in self.test_scores.items():
            test_scores += f"{key}=\n{textwrap.indent(str(value),chr(9))}"
        test_scores = textwrap.indent(test_scores, '\t')
        return f"batchsize={self.batchsize}\nvalidation_scores=\n{validation_scores}\ntest_scores=\n{test_scores}\ntraining_size={self.training_sizes[-1]}"
    
    def to_json(self):
        
        return {"batchsize": self.batchsize, 
                "validation_scores": {key: value.to_json() for key, value in self.validation_scores.items()},
                "test_scores": {key: value.to_json() for key, value in self.test_scores.items()},
                "training_size": self.training_sizes[-1]
                }
 

def make_score(metric, test_labels, individual_predict_labels, committee_predict_labels) -> Score:
    if committee_predict_labels.shape[0] > 0:
        committee_score = metric(test_labels, committee_predict_labels)
        individual_scores = [metric(test_labels, indiv_predict_labels) 
                            for indiv_predict_labels in individual_predict_labels]
        return Score(value=committee_score, minimum=min(individual_scores), maximum=max(individual_scores))
    else:
        return Score(value=numpy.NAN, minimum=numpy.NAN, maximum=numpy.NAN)

def skewed_std_sampling(regressor: BaseCommittee, X: pandas.DataFrame, n_instances: int = 1):
    skew_target = -1
    max_weight = 10
    min_weight = 1
    falloff = 2
    predicts, std = regressor.predict(X, return_std=True) # type: ignore
    weights = numpy.maximum(min_weight, max_weight - numpy.abs(predicts - skew_target) * falloff) * std
    return multi_argmax(weights, n_instances=n_instances)
