from custom_lib import data, ml, stats, vis
from pathlib import Path
import pandas
import numpy as np
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import *
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
from modAL.models import ActiveLearner


train_data = data.load_data(Path('data', 'combined-2025-06-05','samples.csv'))
data.all_augments(train_data)
data.feature_engineer(train_data)
train_data = data.remove_low_filter(train_data)
bootstrap_data = train_data.sample(n=100)

test_data = Path('data', "ppdb_extrapolation",'ppdb_original.csv')
test_data = data.load_data(test_data)
data.all_augments(test_data)
data.feature_engineer(test_data)
test_data_filtered = data.remove_low_filter(test_data)
test_data_critical = data.between_filter(test_data)
test_data_filtered = test_data_filtered.sample(n=min(test_data_filtered.shape[0], 15_000))
test_data_critical = test_data_critical.sample(n=min(test_data_critical.shape[0], 15_000))
test_features, test_labels = ml.split_into_data_and_label(test_data_filtered)
test_features_critical, test_labels_critical = ml.split_into_data_and_label(test_data_critical)



models = {}
kernel = RBF()

model = make_pipeline(StandardScaler(), GaussianProcessRegressor(normalize_y=True))#, kernel=kernel))