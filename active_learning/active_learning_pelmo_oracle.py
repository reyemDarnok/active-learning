import json
import tempfile
import pandas
from sklearn.compose import ColumnTransformer
from custom_lib import data, ml, stats, vis
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Callable, Generator, List, NoReturn, Tuple, Dict

from matplotlib import pyplot as plt

from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.ioTypes.gap import FOCUSCrop, Scenario
from focusStepsPelmo.pelmo.generation_definition import Definition
from focusStepsPelmo.pelmo.local import run_local
from focusStepsPelmo.util.conversions import EnhancedJSONEncoder, flatten, flatten_to_keys

rng = np.random.default_rng()

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.utils._testing import ignore_warnings


from modAL.models import ActiveLearner
from modAL.batch import uncertainty_batch_sampling
from modAL.models import CommitteeRegressor
from modAL.disagreement import max_std_sampling

def transform(X: pandas.DataFrame, y=None):
    local_copy = X.copy()
    data.all_augments(local_copy)
    data.feature_engineer(local_copy)
    return local_copy


def load_dataset(path: Path) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    dataset = data.load_data(path)
    return prep_dataset(dataset)

def prep_dataset(dataset: pandas.DataFrame) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    dataset = data.minimal_filter_raw(dataset)
    return ml.split_into_data_and_label_raw(dataset)

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


test_data = Path(__file__).parent.parent / 'new_ppdb_inferred.csv'

features, labels = load_dataset(Path(__file__).parent.parent /'combined-scan'/'samples.csv')

scratch_space = tempfile.TemporaryDirectory()
scratch_path = Path(scratch_space.name)
output_dir = Path('active_learning')

test_data = data.load_data(test_data)
test_data_filtered = data.remove_low_filter_raw(test_data)
test_data_critical = data.between_filter_raw(test_data)
test_features, test_labels = ml.split_into_data_and_label_raw(test_data_filtered)
test_features_critical, test_labels_critical = ml.split_into_data_and_label_raw(test_data_critical)

model = HistGradientBoostingRegressor()
oneHotEncoder = ColumnTransformer(
    transformers=[
        ('scenario',OneHotEncoder(categories=[[x.name for x in Scenario]]),  ['scenario']),
        ('crop', OneHotEncoder(categories=[[x.name for x in FOCUSCrop]]), ['gap.arguments.modelCrop'])
        ],
    remainder='passthrough'
)
pipeline = make_pipeline(FunctionTransformer(transform),oneHotEncoder, StandardScaler(), model)


bootstrap_size = 10
total_points = 100
batchsize = 20
oversampling_factor = 10
models_in_committee = 1


@ignore_warnings()
def setup_learner(template: Definition):
    learner_list = []
    for index in range(models_in_committee):
        while True:
            combinations, _ = generate_features(template=template, number=bootstrap_size)
            evaluated = evaluate_features(features=combinations)
            features, labels = prep_dataset(evaluated)
            if features.shape[0] != 0:
                break
        committee_member = ActiveLearner(
            estimator=clone(pipeline),
            X_training=features,
            y_training=labels
        )
        learner_list.append(committee_member)
        print(f"created committee member {index} at", datetime.now(), f" with {features.shape[0]} bootstrap points")
    return CommitteeRegressor(
        learner_list = learner_list,
        query_strategy=max_std_sampling, 
    )


custom_metric = stats.make_custom_metric(greater_is_better = False)
false_negative_metric = stats.make_false_negative_metric(greater_is_better = False)
false_positive_metric = stats.make_false_positive_metric(greater_is_better = False)

@ignore_warnings()
def train_learner(learner, batchsize: int, oversampling_factor: float, test_features, test_labels, template: Definition):
    result = ml.TrainingRecord(
        model = learner,
        batchsize = batchsize
    )
    score_time = timedelta(0)
    start_time = datetime.now()
    metrics = {"custom": custom_metric, "false_positive": false_positive_metric, 
               "false_negative": false_negative_metric, "r2": r2_score, "rmse": root_mean_squared_error}
    for name in metrics.keys():
        result.scores[name] = []
    learned_points = 0
    while learned_points < total_points:
        combinations, features = generate_features(template, int(batchsize * oversampling_factor))
        query_ids, _ = learner.query(features, n_instances=batchsize)
        evaluated = evaluate_features(features=list(np.array(combinations)[query_ids]))
        features, labels = prep_dataset(evaluated)
        points_in_batch = features.shape[0]
        if learned_points + points_in_batch > total_points:
            remaining_points = total_points - learned_points
            features = features.iloc[:remaining_points]
            labels = labels.iloc[:remaining_points]
            points_in_batch = remaining_points

        learned_points += points_in_batch # prep dataset removes some points
        # find values for indexes
        if points_in_batch > 0:
            learner.teach(features, labels)
        end_teach = datetime.now() 
        result.training_sizes.append(learned_points)
        result.training_times.append(end_teach - start_time - score_time)
        predict_labels, predict_std = learner.predict(test_features, return_std=True)
        predict_labels = np.ravel(predict_labels)

        individual_predict_labels = [np.ravel(indiv.predict(test_features)) for indiv in learner.learner_list]

        for name, metric in metrics.items():
            committee_score = metric(test_labels, predict_labels)
            individual_scores = [metric(test_labels, indiv_labels) for indiv_labels in individual_predict_labels]
            committee_std = float(np.std(individual_scores))
            result.scores[name].append((committee_score, committee_std))
        end_score = datetime.now()
        score_time += end_score - end_teach
        print(str(result), end_score, end='\r')

    return result
    

def make_training():
    template = Definition.parse(json.loads((Path(__file__).parent / 'scan-matrix-ppdb-based.json').read_text()))

    learner = setup_learner(template=template)
    return train_learner(learner=learner, batchsize=batchsize, oversampling_factor=oversampling_factor, test_features=test_features, test_labels=test_labels, template=template)

records = []
for index in range(1):
    t = make_training()
    print(str(t))
    records.append(t)

record = records[0]

pl_path = Path(__file__).parent / 'plots'
pl_path.mkdir(exist_ok=True, parents=True)
plt.plot( record.training_sizes,[x.total_seconds() for x in record.training_times],)
plt.ylabel("Training time in seconds")
plt.xlabel("Trained Data Points")
plt.savefig(pl_path / 'training_time.svg', bbox_inches='tight')

scores, stds = zip(*record.scores['custom'])
scores = np.array(scores)
plt.plot(record.training_sizes, scores)
plt.fill_between(record.training_sizes, scores - stds, scores + stds)
plt.xlabel("Trained Data Points")
plt.ylabel("custom score")
plt.savefig(pl_path / 'custom_score.svg', bbox_inches='tight')

scores, stds = zip(*record.scores['r2'])
scores = np.array(scores)
plt.plot(record.training_sizes, scores)
plt.fill_between(record.training_sizes, scores - stds, scores + stds)
plt.xlabel("Trained Data Points")
plt.ylabel("r2 score")
plt.ylim(0,1)
plt.savefig(pl_path / 'r2_score.svg', bbox_inches='tight')

scores, stds = zip(*record.scores['false_negative'])
scores = np.array(scores)
plt.plot(record.training_sizes, scores)
plt.fill_between(record.training_sizes, scores - stds, scores + stds)
plt.xlabel("Trained Data Points")
plt.ylabel("false negative score")
plt.ylim(0,1)
plt.savefig(pl_path / 'false_negative_score.svg', bbox_inches='tight')

scores, stds = zip(*record.scores['false_positive'])
scores = np.array(scores)
plt.plot(record.training_sizes, scores)
plt.fill_between(record.training_sizes, scores - stds, scores + stds)
plt.xlabel("Trained Data Points")
plt.ylabel("false positive score")
plt.ylim(0,1)
plt.savefig(pl_path / 'false_positive_score.svg', bbox_inches='tight')

scores, stds = zip(*record.scores['rmse'])
scores = np.array(scores)
plt.plot(record.training_sizes, scores)
plt.fill_between(record.training_sizes, scores - stds, scores + stds)
plt.xlabel("Trained Data Points")
plt.ylabel("rmse score")
plt.ylim(0,1)
plt.savefig(pl_path / 'rmse_score.svg', bbox_inches='tight')


values, stds = record.model.predict(test_features, return_std=True)
values = np.ravel(values)
stds = np.ravel(values)
plt.scatter(test_labels,values, values - stds, values + stds)
plt.plot(test_labels,test_labels,'k-')
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.savefig(pl_path / 'true_v_pred.svg', bbox_inches='tight')

scratch_space.cleanup()