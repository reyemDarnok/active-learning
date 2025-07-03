import json
import math
import tempfile
import numpy
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, NoReturn, Optional, Tuple
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
class TrainingRecord:
    model: BaseCommittee = field(repr=False)
    batchsize: int = 0
    validation_scores: Dict[str, 'ScenarioScores'] = field(default_factory=dict)
    test_scores: Dict[str, 'ScenarioScores'] = field(default_factory=dict)
    training_times: List[timedelta] = field(default_factory=list)
    training_sizes: List[int] = field(default_factory=list)
    all_training_points: Optional[pandas.DataFrame] = None
    usable_points: int = 0
    attempted_points: int = 0
        
    #def __str__(self):
    #    return f"TrainingRecord(batchsize={self.batchsize}, " +\
    #            f"training_time={self.training_times[-1]}, total_points={self.training_sizes[-1]}, scores={ {name: {'value' :f'{scores[-1][0]:2.2}', 'std': f'{scores[-1][1]:2.2}'} for name, scores in self.scores.items()} })"
    
@dataclass
class ScenarioScores:
    combined: List['Score'] = field(default_factory=list)
    scenarios: Dict[Scenario, List['Score']] = field(default_factory=lambda: {s: [] for s in Scenario})

@dataclass
class Score:
    value: float = 0
    std: float = 0