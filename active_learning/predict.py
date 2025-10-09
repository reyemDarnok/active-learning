from zipfile import ZipFile
import joblib
from sklearn.decomposition import PCA
import pandas as pd
from custom_lib.data import prep_dataset, transform
from argparse import ArgumentParser, Namespace
from enum import Enum
from functools import partial
from itertools import product
import json
import logging
from os import cpu_count
import pickle
import random
import tempfile
import math
from uuid import uuid4
import pandas
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from custom_lib import data, ml, stats, vis
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from datetime import datetime, timedelta
from typing import Any, Callable, Generator, List, NoReturn, Optional, Sequence, Tuple, Dict, TypeVar

import matplotlib
from matplotlib import pyplot as plt

from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.ioTypes.gap import FOCUSCrop, Scenario
from focusStepsPelmo.pelmo.generation_definition import Definition
from focusStepsPelmo.pelmo.local import run_local
from focusStepsPelmo.util import jsonLogger
from focusStepsPelmo.util.conversions import EnhancedJSONEncoder, flatten, flatten_to_keys
from active_learning_pelmo_oracle import *

def main():
    args = parse_args_predict()
    model: CommitteeRegressor = load_model(args)
    data = load_data(args)
    result = predict(model, data)
    save_data(args, result)

def save_data(args: Namespace, data: pandas.DataFrame):
    data.to_csv(args.output)

def load_model(args: Namespace) -> CommitteeRegressor:

    with open(args.model, 'rb') as model_file:
        return pickle.load(model_file)

def load_data(args: Namespace) -> pd.DataFrame:
    return pd.read_csv(args.input)



def predict(model: CommitteeRegressor, data: pandas.DataFrame):
    usables = prep_dataset(data)[0]
    prediction = model.predict(usables)
    usables['Prediction'] = prediction
    return usables


def parse_args_predict() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=Path, help='Where to find the model pkl file', default='training/all_crops_all_diagnostics_2/catboost/model.pkl')
    parser.add_argument('-i', '--input', type=Path, help='The input csv file', default=Path('active_learning', 'var_crop.csv'))
    parser.add_argument('-o', '--output', type=Path, help='The output csv file. May be the same as input, overwrites in this case', default=Path('active_learning','pred_result2.csv'))
    return parser.parse_args()

if __name__ == '__main__':
    main()