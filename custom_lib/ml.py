import math
import numpy
from dataclasses import dataclass, field
from typing import Dict, List
from datetime import timedelta


def split_into_data_and_label(dataset):
    #pecs = dataset.columns[dataset.columns.str.endswith('.pec')]
    pecs = ["parent.pec"]
    data = dataset.drop(pecs, axis=1)
    label = dataset[pecs].copy()
    for column in label:
        label[column] = label[column].apply(lambda x: math.log10(x))
    return data, numpy.ravel(label)

def GP_regression_std(regressor, X, n_instances=1):
    _, std = regressor.predict(X, return_std=True)
    idx = numpy.argpartition(std, -n_instances)[-n_instances:]
    return idx, X[idx]

@dataclass
class TrainingRecord:
    model: object = field(repr=False)
    batchsize: int = 0
    scores: Dict[str, List[float]] = field(default_factory=dict)
    training_times: List[timedelta] = field(default_factory=list)
    training_sizes: List[int] = field(default_factory=list)
        
    def __str__(self):
        return f"TrainingRecord(batchsize={self.batchsize}, training_time={self.training_times[-1]}, total_points={self.training_sizes[-1]}, scores={ {name: f'{scores[-1]:2.2}' for name, scores in self.scores.items()} })"