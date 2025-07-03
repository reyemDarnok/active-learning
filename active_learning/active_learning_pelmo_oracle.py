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
from modAL.models.base import BaseCommittee

test_data = Path(__file__).parent.parent / 'new_ppdb_inferred.csv'



test_data = data.load_data(test_data)
#test_data = test_data[test_data['combination.scenarios.0'] == 'C']

test_data_filtered = data.remove_low_filter_raw(test_data)
test_data_critical = data.between_filter_raw(test_data)
test_features, test_labels = ml.split_into_data_and_label_raw(test_data_filtered)
test_features_critical, test_labels_critical = ml.split_into_data_and_label_raw(test_data_critical)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--bootstrap', type=int, default=15, help="How many batches to use to bootstrap the models")
    parser.add_argument('--batch', type=int, default = 8, help="How many batches to run as a group when training")
    parser.add_argument('--total-points', type=int, default=1000, help="How many points to train on")
    parser.add_argument('--oversampling', type=float, default=5, help="Training chooses x points from x*oversampling options")
    parser.add_argument('--models-in-committee', type=int, default=10, help="How many models are in the committee")
    parser.add_argument('--template-path', type=Path, default=Path(__file__).parent / 'scan-matrix-ppdb-based.json', help="Where to find the definition for the scanning template")
    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger()

    jsonLogger.configure_logger_from_argparse(logger, args)
    return args

def main():
    args = parse_args()
    pelmo_batch_size: int = cpu_count() # type: ignore # 15
    bootstrap_size = pelmo_batch_size * args.bootstrap
    total_points = args.total_points
    batchsize = pelmo_batch_size * args.batch
    oversampling_factor = args.oversampling
    models_in_committee = args.models_in_committee
    template = Definition.parse(json.loads(args.template_path.read_text()))


    for _ in range(1):
        learner = setup_learner(template=template, models_in_committee=models_in_committee, bootstrap_size=bootstrap_size)
        t = train_learner(learner=learner, batchsize=batchsize, oversampling_factor=oversampling_factor, validation_features=test_features, validation_labels=test_labels, template=template, total_points=total_points)
        save_training(t)
        print(str(t))


@ignore_warnings()
def setup_learner(template: Definition, models_in_committee: int, bootstrap_size: int) -> BaseCommittee:
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
    ) # type: ignore a base committee fulfills BaseLearners contract, the subclassing is just wrong




@ignore_warnings()
def train_learner(learner: BaseCommittee, 
                  batchsize: int, oversampling_factor: float, 
                  validation_features: pandas.DataFrame, validation_labels: pandas.DataFrame, 
                  template: Definition, total_points: int) -> ml.TrainingRecord:
    custom_metric = stats.make_custom_metric(greater_is_better = False)
    false_negative_metric = stats.make_false_negative_metric(greater_is_better = False)
    false_positive_metric = stats.make_false_positive_metric(greater_is_better = False)

    result = ml.TrainingRecord(
        model = learner,
        batchsize = batchsize
    )
    score_time = timedelta(0)
    start_time = datetime.now()
    metrics = {"custom": custom_metric, "false positive": false_positive_metric, 
               "false negative": false_negative_metric, "R²": r2_score, "RMSE": root_mean_squared_error}
    for name, metric in metrics.items():
        dataset_filters = {
            'total': lambda f,l: pandas.Series([True] *len(f), index=f.index),
            'critical': lambda f,l: l['0.compound_pec'].between(left=1e-2, right= 3, inclusive='neither')
            }
        result.validation_scores[name] = ml.DatasetScores(total_features=validation_features, total_labels=validation_labels, dataset_filters=dataset_filters, metric=metric)
        result.test_scores[name] = ml.DatasetScores(total_features=test_features, total_labels=test_labels, dataset_filters=dataset_filters, metric=metric)
    while result.training_sizes and result.training_sizes[-1] < total_points:
        features, labels = make_training_data(learner=learner, batchsize=batchsize, oversampling_factor=oversampling_factor, template=template, total_points=total_points, result=result)

        # find values for indexes
        if features.shape[0] > 0:
            learner.teach(features, labels)
        end_teach = datetime.now()
        result.training_times.append(end_teach - start_time - score_time)
        
        for scores in result.validation_scores.values():
            scores.update_scores(learner)

        all_features, all_labels = data.prep_dataset(result.all_training_points)
        for scores in result.test_scores.values():
            scores.update_scores(learner, total_features=all_features, total_labels=all_labels)

        end_score = datetime.now()
        score_time += end_score - end_teach
        print(str(result), end_score)
        print(f"usable points: {result.usable_points/result.attempted_points}")

    return result

def make_training_data(learner, batchsize, oversampling_factor, template, total_points, result):
    combinations, features = ml.generate_features(template, int(batchsize * oversampling_factor))
    query_ids, _ = learner.query(features, n_instances=batchsize)
    evaluated = ml.evaluate_features(features=list(np.array(combinations)[query_ids]))
    if result.all_training_points is not None:
        pandas.concat((result.all_training_points, evaluated))
    else:
        result.all_training_points = evaluated
    features, labels = data.prep_dataset(evaluated)
    result.usable_points += features.shape[0]
    result.attempted_points += batchsize
    points_in_batch = features.shape[0]
    if result.training_sizes:
        learned_points = result.training_sizes[-1]
    else:
        learned_points = 0
    if learned_points + points_in_batch > total_points:
        remaining_points = total_points - learned_points
        features = features.iloc[:remaining_points]
        labels = labels.iloc[:remaining_points]
        points_in_batch = remaining_points

    learned_points += points_in_batch # prep dataset removes some points
    result.training_sizes.append(learned_points)
    return features,labels
    
    


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

    visualise_metric(record=record, save_dir=save_dir, metric='custom')
    visualise_metric(record=record, save_dir=save_dir, metric='R²')
    visualise_metric(record=record, save_dir=save_dir, metric='false negative')
    visualise_metric(record=record, save_dir=save_dir, metric='false positive')
    visualise_metric(record=record, save_dir=save_dir, metric='RMSE')


def visualise_metric(record: ml.TrainingRecord, save_dir: Path, metric: str) -> None:
    visualise_validation(record=record, save_dir=save_dir, metric=metric, dataset='total')
    visualise_test(record=record, save_dir=save_dir, metric=metric, dataset='total')
    visualise_validation(record=record, save_dir=save_dir, metric=metric, dataset='critical')
    visualise_test(record=record, save_dir=save_dir, metric=metric, dataset='critical')

    


def visualise_validation(record: ml.TrainingRecord, save_dir: Path, metric: str, dataset: str):
    scenario_scores = record.validation_scores[metric].scores[dataset]
    for scenario in Scenario:
        scores, stds = zip(*[(score.value, score.std) for score in scenario_scores.scenarios[scenario]])
        scores = np.array(scores)
        plt.plot(record.training_sizes, scores, label=f"Scenario {scenario.value}")
        plt.fill_between(record.training_sizes, scores - stds, scores + stds, alpha=0.2)
    scores, stds = zip(*[(score.value, score.std) for score in scenario_scores.combined])
    scores = np.array(scores)
    plt.plot(record.training_sizes, scores, label=f"All Scenarios Combined")
    plt.fill_between(record.training_sizes, scores - stds, scores + stds, alpha=0.2)
    plt.xlabel("Trained Data Points")
    plt.ylabel(f"{metric} score".title())
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f"{metric} Score on {dataset} external validation data".title())
    plt.savefig(save_dir / f'{metric}_{dataset}_score.svg', bbox_inches='tight')
    plt.close('all')


def visualise_test(record: ml.TrainingRecord, save_dir: Path, metric: str, dataset: str):
    scenario_scores = record.test_scores[metric].scores[dataset]
    for scenario in Scenario:
        scores, stds = zip(*[(score.value, score.std) for score in scenario_scores.scenarios[scenario]])
        scores = np.array(scores)
        plt.plot(record.training_sizes, scores, label=f"Scenario {scenario.value}")
        plt.fill_between(record.training_sizes, scores - stds, scores + stds, alpha=0.2)
    scores, stds = zip(*[(score.value, score.std) for score in scenario_scores.combined])
    scores = np.array(scores)
    plt.plot(record.training_sizes, scores, label=f"All Scenarios Combined")
    plt.fill_between(record.training_sizes, scores - stds, scores + stds, alpha=0.2)
    plt.xlabel("Trained Data Points")
    plt.ylabel(f"{metric} score".title())
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f"{metric} Score on {dataset} training data".title())
    plt.savefig(save_dir / f'{metric}_{dataset}_score.svg', bbox_inches='tight')
    plt.close('all')


    

if __name__ == "__main__":
    main()