from argparse import ArgumentParser
import json
import logging
from os import cpu_count
import pickle
import random
import tempfile
from uuid import uuid4
import pandas
from sklearn.compose import ColumnTransformer
from custom_lib import data, ml, stats, vis
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Callable, Generator, List, NoReturn, Optional, Tuple, Dict

from matplotlib import pyplot as plt

from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.ioTypes.gap import FOCUSCrop, Scenario
from focusStepsPelmo.pelmo.generation_definition import Definition
from focusStepsPelmo.pelmo.local import run_local
from focusStepsPelmo.util import jsonLogger
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
from modAL.disagreement import max_std_sampling, max_disagreement_sampling

parser = ArgumentParser()
jsonLogger.add_log_args(parser)
args = parser.parse_args()
logger = logging.getLogger()

jsonLogger.configure_logger_from_argparse(logger, args)

test_data = Path(__file__).parent.parent / 'new_ppdb_inferred.csv'



test_data = data.load_data(test_data)
#test_data = test_data[test_data['combination.scenarios.0'] == 'C']

test_data_filtered = data.remove_low_filter_raw(test_data)
test_data_critical = data.between_filter_raw(test_data)
test_features, test_labels = ml.split_into_data_and_label_raw(test_data_filtered)
test_features_critical, test_labels_critical = ml.split_into_data_and_label_raw(test_data_critical)



def main():
    pelmo_batch_size: int = cpu_count() # type: ignore # 15
    bootstrap_size = pelmo_batch_size * 5
    total_points = 1000
    batchsize = pelmo_batch_size * 4
    oversampling_factor = 5
    models_in_committee = 10
    template = Definition.parse(json.loads((Path(__file__).parent / 'scan-matrix-ppdb-based.json').read_text()))


    for _ in range(1):
        learner = setup_learner(template=template, models_in_committee=models_in_committee, bootstrap_size=bootstrap_size)
        t = train_learner(learner=learner, batchsize=batchsize, oversampling_factor=oversampling_factor, test_features=test_features, test_labels=test_labels, template=template, total_points=total_points)
        save_training(t)
        print(str(t))


@ignore_warnings()
def setup_learner(template: Definition, models_in_committee: int, bootstrap_size: int):
    model = HistGradientBoostingRegressor()
    oneHotEncoder = ColumnTransformer(
        transformers=[
            ('scenario',OneHotEncoder(categories=[[x.name for x in Scenario]]),  ['scenario']),
            ('crop', OneHotEncoder(categories=[[x.name for x in FOCUSCrop]]), ['gap.arguments.modelCrop'])
            ],
        remainder='passthrough'
    )
    pipeline = make_pipeline(FunctionTransformer(data.transform),oneHotEncoder, StandardScaler(), model)

    learner_list = []
    for index in range(models_in_committee):
        while True:
            combinations, _ = ml.generate_features(template=template, number=bootstrap_size)
            evaluated = ml.evaluate_features(features=combinations)
            features, labels = data.prep_dataset(evaluated)
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




@ignore_warnings()
def train_learner(learner, batchsize: int, oversampling_factor: float, test_features, test_labels, template: Definition, total_points: int):
    custom_metric = stats.make_custom_metric(greater_is_better = False)
    false_negative_metric = stats.make_false_negative_metric(greater_is_better = False)
    false_positive_metric = stats.make_false_positive_metric(greater_is_better = False)

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
        start_of_iter = datetime.now()
        print("Starting to generate features", datetime.now() - start_of_iter)
        combinations, features = ml.generate_features(template, int(batchsize * oversampling_factor))
        print("Features generated, starting to select", datetime.now() - start_of_iter)
        query_ids, _ = learner.query(features, n_instances=batchsize)
        print("Features selected, starting to evaluate them", datetime.now() - start_of_iter)
        evaluated = ml.evaluate_features(features=list(np.array(combinations)[query_ids]))
        if result.all_training_points is not None:
            pandas.concat((result.all_training_points, evaluated))
        else:
            result.all_training_points = evaluated
        print("Labels found, starting to prep data", datetime.now() - start_of_iter)
        features, labels = data.prep_dataset(evaluated)
        result.usable_points += features.shape[0]
        result.attempted_points += batchsize
        points_in_batch = features.shape[0]
        if learned_points + points_in_batch > total_points:
            remaining_points = total_points - learned_points
            features = features.iloc[:remaining_points]
            labels = labels.iloc[:remaining_points]
            points_in_batch = remaining_points

        learned_points += points_in_batch # prep dataset removes some points
        # find values for indexes
        print("Data Prep finished, starting to teach", datetime.now() - start_of_iter)
        if points_in_batch > 0:
            learner.teach(features, labels)
        end_teach = datetime.now()
        print("Finished teaching, starting to evaluate metrics", datetime.now() - start_of_iter) 
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
        print(str(result), end_score)
        print(f"usable points: {result.usable_points/result.attempted_points}({result.usable_points}/{result.attempted_points})")

    return result
    
    


def save_training(record: ml.TrainingRecord, save_dir: Path = Path(__file__).parent / f'training{str(uuid4())}'):
    save_dir.mkdir(exist_ok=True, parents=True)
    if record.all_training_points is not None:
        record.all_training_points.to_csv(save_dir / 'training_data.csv')
    print(f"Writing results to {save_dir}")
    with (save_dir / "committee.pickle").open('wb') as picklefile:
        pickle.dump(record, picklefile)
    plt.plot( record.training_sizes,[x.total_seconds() for x in record.training_times],)
    plt.ylabel("Training time in seconds")
    plt.xlabel("Trained Data Points")
    plt.savefig(save_dir / 'training_time.svg', bbox_inches='tight')
    plt.close('all')

    scores, stds = zip(*record.scores['custom'])
    scores = np.array(scores)
    plt.plot(record.training_sizes, scores)
    plt.fill_between(record.training_sizes, scores - stds, scores + stds, alpha=0.2)
    plt.xlabel("Trained Data Points")
    plt.ylabel("custom score")
    plt.savefig(save_dir / 'custom_score.svg', bbox_inches='tight')
    plt.close('all')


    scores, stds = zip(*record.scores['r2'])
    scores = np.array(scores)
    plt.plot(record.training_sizes, scores)
    plt.fill_between(record.training_sizes, scores - stds, scores + stds, alpha=0.2)
    plt.xlabel("Trained Data Points")
    plt.ylabel("r2 score")
    plt.ylim(top=1)
    plt.savefig(save_dir / 'r2_score.svg', bbox_inches='tight')
    plt.close('all')


    scores, stds = zip(*record.scores['false_negative'])
    scores = np.array(scores)
    plt.plot(record.training_sizes, scores)
    plt.fill_between(record.training_sizes, scores - stds, scores + stds, alpha=0.2)
    plt.xlabel("Trained Data Points")
    plt.ylabel("false negative score")
    plt.ylim(0,1)
    plt.savefig(save_dir / 'false_negative_score.svg', bbox_inches='tight')
    plt.close('all')


    scores, stds = zip(*record.scores['false_positive'])
    scores = np.array(scores)
    plt.plot(record.training_sizes, scores)
    plt.fill_between(record.training_sizes, scores - stds, scores + stds, alpha=0.2)
    plt.xlabel("Trained Data Points")
    plt.ylabel("false positive score")
    plt.ylim(0,1)
    plt.savefig(save_dir / 'false_positive_score.svg', bbox_inches='tight')
    plt.close('all')


    scores, stds = zip(*record.scores['rmse'])
    scores = np.array(scores)
    plt.plot(record.training_sizes, scores)
    plt.fill_between(record.training_sizes, scores - stds, scores + stds, alpha=0.2)
    plt.xlabel("Trained Data Points")
    plt.ylabel("rmse score")
    plt.ylim(bottom=0)
    plt.savefig(save_dir / 'rmse_score.svg', bbox_inches='tight')
    plt.close('all')


    values, stds = record.model.predict(test_features, return_std=True)
    values = np.ravel(values)
    stds = np.ravel(values)
    plt.scatter(test_labels,values, values - stds, values + stds)
    plt.plot(test_labels,test_labels,'k-')
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.ylim(test_labels.iloc[:, 0].min()*1.1,test_labels.iloc[:, 0].max()*1.1)
    plt.savefig(save_dir / 'true_v_pred.svg', bbox_inches='tight')
    plt.close('all')

if __name__ == "__main__":
    main()