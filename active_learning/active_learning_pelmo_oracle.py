import json
import tempfile
import pandas
from custom_lib import data, ml, stats, vis
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from typing import Any, List, Tuple, Dict

from matplotlib import pyplot as plt

from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.ioTypes.gap import Scenario
from focusStepsPelmo.pelmo.generation_definition import Definition
from focusStepsPelmo.pelmo.local import run_local
from focusStepsPelmo.util.conversions import EnhancedJSONEncoder, flatten, flatten_to_keys

rng = np.random.default_rng()

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.utils._testing import ignore_warnings


from modAL.models import ActiveLearner
from modAL.batch import uncertainty_batch_sampling
from modAL.models import CommitteeRegressor
from modAL.disagreement import max_std_sampling

def transform(X: pandas.DataFrame, y=None):
    X = X.copy()
    data.all_augments(X)
    data.feature_engineer(X)
    return X


def load_dataset(path: Path):
    dset = data.load_data(path)
    dset = data.remove_low_filter_raw(dset)
    return ml.split_into_data_and_label_raw(dset)


def generate_features(template: Definition, number: int):
    combinations: List[Combination] = [Combination(**template.make_sample()) for _ in range(number)] 
    flattened = [list(flatten(combination.asdict())) for combination in combinations]
    return combinations, pandas.DataFrame(flattened, columns=list(flatten_to_keys(combinations[0].asdict())))

def evaluate_features(features: List[Combination]):
    feature_tuple = tuple(features)
    name = hash(feature_tuple)
    with tempfile.TemporaryDirectory() as work:
        work_dir = Path(work)
        combination_path = (work_dir / 'combination' / f"{name}.json")
        with combination_path.open('w') as combination_file:
            json.dump(feature_tuple, fp=combination_file, cls=EnhancedJSONEncoder)
        pelmo_res_path = output_dir / 'pelmo' / f"{name}.csv"
        run_local(work_dir=work_dir / 'pelmo', output_file=pelmo_res_path,combination_dir=combination_path,scenarios=frozenset([Scenario.C]))
        result_df = pandas.read_csv(pelmo_res_path)
        return result_df


test_data = Path(__file__).parent.parent /'ppdb'/'inferred.csv'
features, labels = load_dataset(Path(__file__).parent.parent /'combined-scan'/'samples.csv')
scratch_space = tempfile.TemporaryDirectory()
scratch_path = Path(scratch_space.name)
output_dir = Path('active_learning')

test_data = data.load_data(test_data)
test_data_filtered = data.remove_low_filter_raw(test_data)
test_data_critical = data.between_filter_raw(test_data)
test_features, test_labels = ml.split_into_data_and_label_raw(test_data_filtered)
test_features_critical, test_labels_critical = ml.split_into_data_and_label_raw(test_data_critical)

kernel = RBF()
model = GaussianProcessRegressor(normalize_y=True)
pipeline = make_pipeline(FunctionTransformer(transform), StandardScaler(), model)


bootstrap_size = 150
total_points = 1000
batchsize = 20
oversampling_factor = 3.5

@ignore_warnings()
def setup_learner() -> ActiveLearner:
    bootstrap_index = rng.choice(features.shape[0], bootstrap_size, replace=False)
    bootstrap_features = features.iloc[bootstrap_index]
    bootstrap_labels = labels.iloc[bootstrap_index] # type: ignore
    learner = ActiveLearner(
        estimator = clone(pipeline),
        query_strategy=ml.GP_regression_std, 
        X_training=bootstrap_features,
        y_training=bootstrap_labels
    )
    return learner

custom_metric = stats.make_custom_metric(greater_is_better = False)
false_negative_metric = stats.make_false_negative_metric(greater_is_better = False)
false_positive_metric = stats.make_false_positive_metric(greater_is_better = False)

@ignore_warnings()
def train_learner(learner, batchsize: int, oversampling_factor: float, test_features, test_labels, template_path: Path) -> ml.TrainingRecord:
    result = ml.TrainingRecord(
        model = learner,
        batchsize = batchsize
    )
    score_time = timedelta(0)
    start_time = datetime.now()
    metrics = {"custom": custom_metric, "false_positive": false_positive_metric, 
               "false_negative": false_negative_metric, "r2": r2_score}
    for name in metrics.keys():
        result.scores[name] = []
    sample_definition = Definition.parse(json.loads(template_path.read_text()))
    for i in range(bootstrap_size, total_points, batchsize):
        combinations, features = generate_features(sample_definition, int(batchsize * oversampling_factor))
        query_ids, selected_features = learner.query(features, n_instances=min(batchsize, total_points - i))
        result = evaluate_features(features=list(np.array(combinations)[query_ids]))
        result = data.remove_low_filter_raw(features)
        features, labels = ml.split_into_data_and_label_raw(features)
        # find values for indexes
        learner.teach(selected_features, labels)
        end_teach = datetime.now() 
        result.training_sizes.append(learner.X_training.shape[0])
        result.training_times.append(end_teach - start_time - score_time)
        predict_labels = learner.predict(test_features)
        predict_labels = np.ravel(predict_labels)
       
        for name, metric in metrics.items():
            result.scores[name].append(metric(test_labels, predict_labels))
        end_score = datetime.now()
        score_time += end_score - end_teach
        print(str(result), end_score, end='\r')

    return result
    

def make_training():
    learner = setup_learner()
    return train_learner(learner=learner, batchsize=batchsize, oversampling_factor=oversampling_factor, test_features=test_features, test_labels=test_labels, template_path=Path(__file__).parent / 'scan-matrix-ppdb-based.json')

records = []
for index in range(1):
    t = make_training()
    print(str(t))
    records.append(t)

record = records[0]

plt.plot( record.training_sizes,[x.total_seconds() for x in record.training_times],)
plt.ylabel("Training time in seconds")
plt.xlabel("Trained Data Points")
#plt.axis([x1, x2, y1, y2])

plt.show()

plt.plot(record.training_sizes, record.scores['custom'])
plt.xlabel("Trained Data Points")
plt.ylabel("custom score")
plt.show()

plt.plot(record.training_sizes, record.scores['r2'])
plt.xlabel("Trained Data Points")
plt.ylabel("r2 score")
plt.ylim(0,1)
plt.show()

plt.plot(record.training_sizes, record.scores['false_negative'])
plt.xlabel("Trained Data Points")
plt.ylabel("false negative score")
plt.ylim(0,1)
plt.show()

plt.plot(record.training_sizes, record.scores['false_positive'])
plt.xlabel("Trained Data Points")
plt.ylabel("false positive score")
plt.ylim(0,1)
plt.show()

plt.scatter(test_labels,record.model.predict(test_features))
plt.plot(test_labels,test_labels,'k-')
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.ylim(-4,4)
plt.show()

scratch_space.cleanup()