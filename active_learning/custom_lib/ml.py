import json
import math
import tempfile
import numpy
import numpy.typing as npt
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, NoReturn, Optional, Tuple
from datetime import timedelta
from pathlib import Path

import pandas
from sys import path

from focusStepsPelmo.ioTypes.scenario import Scenario
from focusStepsPelmo.pelmo.local import run_local
from focusStepsPelmo.util.conversions import EnhancedJSONEncoder, flatten, flatten_to_keys
path.append(str(Path(__file__).parent.parent.parent))
from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.pelmo.generation_definition import Definition
from modAL.models.base import BaseCommittee


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
    combinations: List[Combination] = [next(combination_gen) for _ in range(number)] 
    flattened = [list(flatten({'combination': combination.asdict()})) for combination in combinations]
    return combinations, pandas.DataFrame(flattened, columns=list(flatten_to_keys({'combination': combinations[0].asdict()})))

def _combination_generator(template: Definition) -> Generator[Combination, Any, NoReturn]:
    while True:
        try:
            yield Combination(**template.make_sample())
        except ValueError:
            pass

def evaluate_features(features: List[Combination]):
    feature_tuple = tuple(features)
    name = hash(feature_tuple)
    with tempfile.TemporaryDirectory() as work:
        work_dir = Path(work)
        combination_path = (work_dir / 'combination' / f"{name}.json")
        combination_path.parent.mkdir(exist_ok=True, parents=True)
        with combination_path.open('w') as combination_file:
            json.dump(feature_tuple, fp=combination_file, cls=EnhancedJSONEncoder)
        pelmo_res_path = work_dir / 'pelmo_result' / f"{name}.csv"
        run_local(output_file=pelmo_res_path,combination_dir=combination_path,scenarios=frozenset([Scenario.C]))
        result_df = pandas.read_csv(pelmo_res_path)
        return result_df

       
    
@dataclass
class Score:
    value: float = 0
    std: float = 0

@dataclass
class ScenarioScores:
    metric: Callable
    combined: List['Score'] = field(default_factory=list)
    scenarios: Dict[Scenario, List['Score']] = field(default_factory=lambda: {s: [] for s in Scenario})
    
    def __repr__(self):
        return f"ScenarioScores(combined={self.combined[-1] if self.combined else numpy.NAN})"
        

    def update_scores(self, learner: BaseCommittee, test_labels: pandas.DataFrame, test_features: pandas.DataFrame) -> None:
        if test_features.shape[0] > 0:
            individual_predict_labels = [indiv.predict(test_features) for indiv in learner.learner_list]
            committee_predict_labels = learner.predict(test_features)
        else:
            individual_predict_labels = [numpy.ndarray((0,1)) for _ in learner.learner_list]
            committee_predict_labels = numpy.ndarray((0,1))
        for scenario in test_features['combination.scenarios.0'].unique():
            scenario_index = test_features['combination.scenarios.0'] == scenario
            self.scenarios[Scenario[scenario]].append(
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

    def __repr__(self):
        return str(self.scores)

    def __post_init__(self):
        assert self.total_features.shape[0] == self.total_labels.shape[0]
        self._filter()

    def _filter(self):
        for name, filter in self.dataset_filters.items():
            index = filter(self.total_features, self.total_labels)
            self._filtered_features[name] = self.total_features[index]
            self._filtered_labels[name] = self.total_labels[index]
            self.scores[name] = ScenarioScores(metric=self.metric)

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
 
def make_score(metric, test_labels, individual_predict_labels, committee_predict_labels) -> Score:
    if committee_predict_labels.shape[0] > 0:
        committee_score = metric(test_labels, committee_predict_labels)
        individaul_scores = [metric(test_labels, indiv_predict_labels) 
                            for indiv_predict_labels in individual_predict_labels]
        committee_std = float(numpy.std(individaul_scores))
        return Score(value=committee_score, std=committee_std)
    else:
        return Score(value=numpy.NAN, std=numpy.NAN)

def __repr__(self):
    return f"TraininingRecord({self.batchsize=}, validation_scores={self.validation_scores!r}, test_scores={self.test_scores!r}, training_sizes={self.training_sizes[-1]})"