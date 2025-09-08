from sklearn.decomposition import PCA
import pandas as pd
from custom_lib.data import prep_dataset, transform
from argparse import ArgumentParser
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


with open('training/all_crops_all_diagnostics_2/catboost/model.pkl', 'rb') as model_file:
    model: CommitteeRegressor = pickle.load(model_file)
test_set = prep_dataset(pd.read_csv('active_learning/var_crop.csv'))

predictions = model.predict(test_set[0])
lowest = min(*predictions, *np.ravel(test_set[1]))
highest = max(*predictions, *np.ravel(test_set[1]))
plt.scatter(predictions, np.ravel(test_set[1]), s=0.2)
plt.plot([lowest, highest], [lowest, highest], color='red')
plt.xlabel('True Values')
plt.ylabel('Model Predictions')
plt.title('Performance of the Model on the Test Set')
plt.savefig('active_learning/test_predict_old.svg', bbox_inches='tight')
train_set = prep_dataset(pd.read_csv('training/all_crops_all_diagnostics/results/training_data.csv'))
