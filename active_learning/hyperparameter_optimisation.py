from pathlib import Path

import pandas
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from custom_lib.ml import PartialOneHot
from custom_lib import ml
from active_learning_pelmo_oracle import mechanistic_preprocessing
from custom_lib import data

model = xgboost.XGBRegressor(
    #### BEGIN XGBOOST PARAMS
    n_estimators=603,
    max_depth=3,
    max_leaves=0,
    max_bin=162,
    gamma=0.002,
    reg_alpha = 0.041,
    reg_lambda=82.499,
    base_score=-1,

    #### END   XGBOOST PARAMS
)
cat_features = ['scenario', 'gap.arguments.modelCrop']

preprocessing = make_pipeline(FunctionTransformer(data.transform), FunctionTransformer(mechanistic_preprocessing),
                         ml.PartialScaler(to_exclude=cat_features, copy=False), PartialOneHot(to_encode=cat_features),
                         )


training_data = data.load_data(Path('../training/bs16ba16tp5000/results/training_data.csv'))
training_data = data.remove_low_filter_raw(training_data)
training_features, training_labels = ml.split_into_data_and_label_raw(training_data)
training_features = preprocessing.fit_transform(training_features)


params = {
    'reg_lambda': [i / 1000  for i in range(75000,90000, 1)],
}
grid_search = GridSearchCV(model, params, verbose=3, n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(training_features, training_labels)
print(grid_search.best_params_)
Path('grid_search_results_6').write_text(grid_search.best_params_.__str__() + "\n" + params.__str__())

