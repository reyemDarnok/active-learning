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
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression



from custom_lib.ml import PartialOneHot
from custom_lib import data, ml, stats, vis
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from datetime import datetime, timedelta
from typing import Any, Callable, Generator, List, NoReturn, Optional, Sequence, Tuple, Dict, TypeVar

import matplotlib
matplotlib.use('Agg')
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
from sklearn.base import BaseEstimator, clone
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

from catboost import CatBoostRegressor

test_data = Path(__file__).parent / 'var_crop.csv'



test_data = data.load_data(test_data)
#test_data = test_data[test_data['combination.scenarios.0'] == 'C']

test_data_filtered = data.remove_low_filter_raw(test_data)
test_data_critical = data.between_filter_raw(test_data)
test_features, test_labels = ml.split_into_data_and_label_raw(test_data_filtered)
test_features_critical, test_labels_critical = ml.split_into_data_and_label_raw(test_data_critical)

def mechanistic_preprocessing(X: pandas.DataFrame, y=None):
    X.columns
    return X


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--bootstrap', type=int, default=1, help="How many batches to use to bootstrap the models")
    parser.add_argument('--batch', type=int, default = 1, help="How many batches to run as a group when training")
    parser.add_argument('--total-points', type=int, default=1000, help="How many points to train on")
    parser.add_argument('--oversampling', type=float, default=20, help="Training chooses x points from x*oversampling options")
    parser.add_argument('--models-in-committee', type=int, default=10, help="How many models are in the committee")
    parser.add_argument('--template-path', type=Path, default=Path(__file__).parent / 'scan-matrix-ppdb-based_all_crops.json', help="Where to find the definition for the scanning template")
    parser.add_argument('--name', type=str, default=str(uuid4()), help="The name to use for documenting this run")
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).parent.parent / "training", help="Where to write the results to")
    parser.add_argument('--catboost-iter', type=int, default=10_000, help="How many catboost iterations to do")
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


    pkl_file = args.results_dir / args.name / 'catboost' / 'model.pkl'
    if pkl_file.exists():
        with pkl_file.open('rb') as pkl:
            learner = pickle.load(pkl)
    else:
        learner = setup_learner(template=template, models_in_committee=models_in_committee, bootstrap_size=bootstrap_size, name=args.name, result_dir=args.results_dir, catboost_iter=args.catboost_iter)
    t = train_learner(learner=learner, batchsize=batchsize,
                        oversampling_factor=oversampling_factor, 
                        validation_features=test_features, validation_labels=test_labels,
                            template=template, total_points=total_points,
                            save_name=args.name, save_dir=args.results_dir)
    print(str(t))

def make_pd(X, y=None):
    res =  pandas.DataFrame(X, columns=['parent.dt50.sediment', 'parent.dt50.soil', 'parent.dt50.surfaceWater', 'parent.dt50.system', 'parent.freundlich', 'parent.henry', 'parent.koc', 'parent.molarMass', 'parent.plant_uptake', 'parent.reference_temperature', 'parent.vapor_pressure', 'parent.water_solubility', 'gap.arguments.apply_every_n_years', 'gap.arguments.bbch', 'gap.arguments.interval', 'gap.arguments.modelCrop', 'gap.arguments.number_of_applications', 'gap.arguments.rate', 'gap.arguments.season', 'scenario', 'parent.log_henry'])
    return res

def one_hot_encoding(X: pandas.DataFrame, y=None):
    cat_columns = ['scenario', 'gap.arguments.modelCrop']
    cat_frame = X[cat_columns]

    pandas.concat([X, X.get_dummies(columns=cat_columns)], axis=1)
    X.drop(columns=cat_columns, inplace=True)
    return X

@ignore_warnings()
def setup_learner(template: Definition, models_in_committee: int, bootstrap_size: int, name: str, result_dir: Path, catboost_iter: int) -> BaseCommittee:
   

    learner_list = []
    catboost_base = result_dir / name / "catboost"
    catboost_base.mkdir(parents=True, exist_ok=True)
    print(catboost_base.absolute())
    for index in range(models_in_committee):
        cat_features = ['scenario', 'gap.arguments.modelCrop']
        cpus_available = cpu_count() - 1 if cpu_count() is not None else 1 # type: ignore
        metrics = [ "R2", "RMSE"]
        model = xgboost.XGBRegressor(
            #### BEGIN XGBOOST PARAMS
            #n_estimators=100,
            #max_depth=50,
            #max_leaves=0,
            #max_bin=100,
            #gamma=0.01,
            #reg_lambda=0.2,
            base_score=-1,

            #### END   XGBOOST PARAMS
        )

        pipeline = make_pipeline(FunctionTransformer(data.transform), FunctionTransformer(mechanistic_preprocessing), ml.PartialScaler(to_exclude=cat_features, copy=False), PartialOneHot(to_encode=cat_features, copy=False), model)
        while True:
            combinations, _ = ml.generate_features(template=template, number=bootstrap_size)
            evaluated = ml.evaluate_features(features=combinations)
            features, labels = data.prep_dataset(evaluated)
            if features.shape[0] != 0:
                break
        committee_member = ActiveLearner(
            estimator=pipeline,
            X_training=features,
            y_training=labels
        )
        committee_member.bootstrap_evaluated = evaluated # type: ignore
        learner_list.append(committee_member)
        print(f"created committee member {index} at", datetime.now(), f" with {features.shape[0]} bootstrap points")
    return CommitteeRegressor(
        learner_list = learner_list,
        query_strategy=ml.skewed_std_sampling, 
    ) # type: ignore 


def true_index(features, labels):
    return pandas.Series([True] *len(features), index=features.index)

def between_index(features, labels):
    return labels['0.compound_pec'].between(left=1e-2, right= 3, inclusive='neither')



@ignore_warnings()
def train_learner(learner: BaseCommittee, 
                  batchsize: int, oversampling_factor: float, 
                  validation_features: pandas.DataFrame, validation_labels: pandas.DataFrame, 
                  template: Definition, total_points: int, save_dir: Path, save_name: str) -> ml.TrainingRecord:
    custom_metric = stats.make_custom_metric(greater_is_better = False)
    custom_rmse_metric = stats.make_custom_rmse_metric(greater_is_better= False)
    false_negative_metric = stats.make_false_negative_metric(greater_is_better = False)
    false_positive_metric = stats.make_false_positive_metric(greater_is_better = False)
    long_jump_metric = stats.make_long_jump_metric(greater_is_better=False)
    up_jump_metric = stats.make_up_jump_metric(greater_is_better=False)
    down_jump_metric = stats.make_down_jump_metric(greater_is_better=False)


    result = ml.TrainingRecord(
        model = learner,
        batchsize = batchsize
    )
    score_time = timedelta(0)
    start_time = datetime.now()
    metrics = {"custom": custom_metric, "false positive": false_positive_metric, 
               "false negative": false_negative_metric, "R²": r2_score, 
               "RMSE": root_mean_squared_error,
               "Custom RMSE": custom_rmse_metric,
               "Long jumps": long_jump_metric,
               "Upwards jumps": up_jump_metric,
               "Downwards jumps": down_jump_metric,
               "PEC std": partial(stats.pec_std_metric, greater_is_better=False),
               "PEC Intervall": partial(stats.pec_interval_metric, greater_is_better=False)}

    dataset_filters = {
        'total': true_index,
        'critical': between_index
        }
    for name, metric in metrics.items():

        result.scores[ml.Category.CONFIRM].dataset_scores[name] = ml.DatasetScores(total_features=validation_features, total_labels=validation_labels, dataset_filters=dataset_filters, metric=metric)
        result.scores[ml.Category.TRAIN].dataset_scores[name] = ml.DatasetScores(total_features=test_features, total_labels=test_labels, dataset_filters=dataset_filters, metric=metric)
    old_record = save_dir / 'results' / 'record.json'
    if old_record.exists():
        with old_record.open() as old:
            result.load_old_record(json.load(old))
    while not result.training_sizes or result.training_sizes[-1] < total_points:
        features, labels = make_training_data(learner=learner, batchsize=batchsize, oversampling_factor=oversampling_factor, template=template, total_points=total_points, result=result)

        # find values for indexes
        if features.shape[0] > 0:
            learner.teach(features, labels)
        end_teach = datetime.now()
        result.training_times.append(end_teach - start_time - score_time)
        
        for scores in result.scores[ml.Category.CONFIRM].dataset_scores.values():
            scores.update_scores(learner)

        all_features, all_labels = data.prep_dataset(result.all_training_points)
        for scores in result.scores[ml.Category.TRAIN].dataset_scores.values():
            scores.update_scores(learner, total_features=all_features, total_labels=all_labels)

        end_score = datetime.now()
        score_time += end_score - end_teach
        save_training(result, save_dir=save_dir, save_name=save_name, validation_features=validation_features, validation_labels=validation_labels)
        print(result.training_sizes[-1], datetime.now())

    return result

def make_training_data(learner, batchsize, oversampling_factor, template, total_points, result):
    combinations, features = ml.generate_features(template, int(batchsize * oversampling_factor))
    query_ids, _ = learner.query(features, n_instances=batchsize)
    evaluated = ml.evaluate_features(features=combinations[query_ids])
    if result.all_training_points is not None:
        result.all_training_points = pandas.concat((result.all_training_points, evaluated))
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
    
    


def save_training(record: ml.TrainingRecord, save_name: str, save_dir: Path, validation_features, validation_labels):
    catboost_dir = save_dir / save_name / "catboost"
    save_dir = save_dir / save_name / "results"
    save_dir.mkdir(exist_ok=True, parents=True)
    if record.all_training_points is not None:
        record.all_training_points.to_csv(save_dir / 'training_data.csv')
    print(f"Writing results to {save_dir}")
    with (catboost_dir / "model.pkl").open('wb') as picklefile:
        pickle.dump(record.model, picklefile)
    with open(save_dir / "record.json", 'w') as json_out:
        json.dump(record.to_json(), json_out)
    
    visualise_predictions(model=record.model, validation_features=validation_features, validation_labels=validation_labels, save_dir=save_dir, category=ml.Category.CONFIRM)
    training_features, training_labels = data.prep_dataset(record.all_training_points)
    visualise_predictions(model=record.model, validation_features=training_features, validation_labels=training_labels, save_dir=save_dir, category=ml.Category.TRAIN)
    plt.plot( record.training_sizes,[x.total_seconds() for x in record.training_times],)
    plt.ylabel("Training time in seconds")
    plt.xlabel("Trained Data Points")
    plt.savefig(save_dir / 'training_time.svg', bbox_inches='tight')
    plt.close('all')

    for metric in record.scores[ml.Category.TRAIN].dataset_scores.keys():
        visualise_metric(record=record, save_dir=save_dir, metric=metric)

def visualise_predictions(model, validation_features, validation_labels, save_dir: Path, category: ml.Category):
    plot_dir = save_dir / 'predictions' / category.value
    plot_dir.mkdir(exist_ok=True, parents=True)
    predictions = model.predict(validation_features)
    lowest = min(*predictions, *np.ravel(validation_labels))
    highest = max(*predictions, *np.ravel(validation_labels))
    plt.scatter(np.ravel(validation_labels), predictions, s=0.2)
    plt.plot([lowest, highest], [lowest, highest], color='red')
    plt.xlabel('True Values')
    plt.ylabel('Model Predictions')
    plt.title(f'Performance of the Model on the {category.value} Set')
    plt.savefig(plot_dir / 'all.svg', bbox_inches='tight')
    plt.close('all')
    for index, scenario in enumerate(Scenario):
        scenario_index: pandas.Series = (validation_features['combination.scenarios.0'] == scenario.name) | (validation_features['combination.scenarios.0'] == index) 
        if scenario_index.any():
            predictions = model.predict(validation_features[scenario_index])
            truth = validation_labels[scenario_index]
            plt.scatter(truth, predictions, s=0.5)
            plt.plot([lowest, highest], [lowest, highest], color='red')
            plt.xlabel('True Values')
            plt.ylabel('Model Predictions')
            plt.title(f'Performance of the Model on the {category.value} Set')
            plt.savefig(plot_dir / f'{scenario.name}.svg', bbox_inches='tight')
            plt.close('all')

def visualise_metric(record: ml.TrainingRecord, save_dir: Path, metric: str) -> None:
    datasets = next(iter(record.scores[ml.Category.TRAIN].dataset_scores.values())).scores.keys()
    for dataset, category in product(datasets, list(ml.Category)):
        visualise_metric_dataset(record=record, save_dir=save_dir, metric=metric, dataset=dataset, category=category)
        visualise_model_diagnostics(record=record, save_dir=save_dir, metric=metric, dataset=dataset, category=category)
   

def visualise_std_diagnostic(scenario_scores: ml.ScenarioScores, training_sizes: List[int], selection_path: Path, metric: str, dataset: str, category: ml.Category):
    for scenario in Scenario:
        if scenario_scores.scenarios[scenario]:
            std, _ = zip(*[(score.std, score.maximum - score.minimum) for score in scenario_scores.scenarios[scenario]])
            plt.plot(training_sizes, std, label=f"Scenario {scenario.value} Standard Deviation")

    std, _ = zip(*[(score.std, score.maximum - score.minimum) for score in scenario_scores.combined])
    plt.plot(training_sizes, std, label=f"All Scenarios Combined Standard Deviation")

    plt.title(f"{metric} Standard Deviation Diagnostics on {dataset} {category.value} data".title())
    plt.xlabel("Trained Data Points")
    plt.ylabel(f"{metric} score".title())
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(selection_path /  "std.svg", bbox_inches='tight')
    plt.close('all')
    
    plt.plot(training_sizes, std, label=f"All Scenarios Combined Standard Deviation")

    plt.title(f"{metric} Standard Deviation Diagnostics on {dataset} {category.value} data".title())
    plt.xlabel("Trained Data Points")
    plt.ylabel(f"{metric} score".title())
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(selection_path / "std_only_combined.svg", bbox_inches='tight')
    plt.close('all')

def visualise_interval_diagnostic(scenario_scores: ml.ScenarioScores, training_sizes: List[int], selection_path: Path, metric: str, dataset: str, category: ml.Category):
    for scenario in Scenario:
        if scenario_scores.scenarios[scenario]:
            _, diff = zip(*[(score.std, score.maximum - score.minimum) for score in scenario_scores.scenarios[scenario]])
            plt.plot(training_sizes, diff, label=f"Scenario {scenario.value} Prediction Interval")

    _, diff = zip(*[(score.std, score.maximum - score.minimum) for score in scenario_scores.combined])
    plt.plot(training_sizes, diff, label=f"All Scenarios Combined Prediction Interval")
    plt.title(f"{metric} Prediction Interval Diagnostics on {dataset} {category.value} data".title())
    plt.xlabel("Trained Data Points")
    plt.ylabel(f"{metric} score".title())
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(selection_path / 'interval.svg', bbox_inches='tight')
    plt.close('all')
    
    plt.plot(training_sizes, diff, label=f"All Scenarios Combined Prediction Interval")

    plt.title(f"{metric} Prediction Interval Diagnostics on {dataset} {category.value} data".title())
    plt.xlabel("Trained Data Points")
    plt.ylabel(f"{metric} score".title())
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(selection_path / 'interval_only_combined.svg', bbox_inches='tight')
    plt.close('all')

T = TypeVar('T')

def walk_slices(sequence: NDArray, length: int) -> Generator[NDArray, None, None]:
    for i in range(length, len(sequence)):
        yield sequence[i-length:i]

def visualise_slope_diagnostics(scenario_scores: ml.ScenarioScores, training_sizes: List[int], selection_path: Path, metric: str, dataset: str, category: ml.Category, slope_length: int):
    regression = LinearRegression()
    for scenario in Scenario:
        curr_scores = np.array([score.value for score in scenario_scores.scenarios[scenario]])
        training_copy = np.array(training_sizes)
        nan_index = np.isnan(curr_scores)
        curr_scores = curr_scores[~nan_index]
        training_copy = training_copy[~nan_index] 
        if curr_scores.shape[0] > 1:
            points = [(regression.fit(x.reshape(-1,1),y.reshape(-1,1)), regression.coef_[0])[1] 
                      for x, y in zip(walk_slices(training_copy, slope_length), 
                                      walk_slices(curr_scores,slope_length))]
            plt.plot(training_copy[slope_length:], points, label=f"Scenario {scenario.value} Best Slope for last {slope_length} scores".title())

    curr_scores = np.array([score.value for score in scenario_scores.combined])
    training_copy = np.array(training_sizes)
    nan_index = np.isnan(curr_scores)
    curr_scores = curr_scores[~nan_index]
    training_copy = training_copy[~nan_index] 
    if curr_scores.shape[0] > 1:
        points = [(regression.fit([[value] for value in x],y), regression.coef_[0])[1] 
                    for x, y in zip(walk_slices(training_copy, slope_length), # type: ignore
                                    walk_slices(curr_scores,slope_length))] # type: ignore    
        plt.plot(training_copy[slope_length:], points, label=f"All Scenarios Combined Best Slope for last {slope_length} scores".title())
        plt.plot([training_copy[0], training_copy[-1]], [0,0], color='red')
        plt.title(f"{metric} Best Slope for last {slope_length} Diagnostics on {dataset} {category.value} data".title())
        plt.xlabel("Trained Data Points")
        plt.ylabel(f"{metric} score slope".title())
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig(selection_path / f'slope_{slope_length}.svg', bbox_inches='tight')
        plt.close('all')

        plt.plot(training_copy[slope_length:], points, label=f"All Scenarios Combined Best Slope for last {slope_length} scores".title())
        plt.plot([training_copy[0], training_copy[-1]], [0,0], color='red')
        plt.title(f"{metric} Best Slope for last {slope_length} Diagnostics on {dataset} {category.value} data".title())
        plt.xlabel("Trained Data Points")
        plt.ylabel(f"{metric} score slope".title())
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig(selection_path / f'slope_{slope_length}_only_combined.svg', bbox_inches='tight')
        plt.close('all')

def visualise_progression_diagnostics(scenario_scores: ml.ScenarioScores, training_sizes: List[int], selection_path: Path, metric: str, dataset: str, category: ml.Category):
    for scenario in Scenario:
        if scenario_scores.scenarios[scenario]:
            numpy_scores = np.array([s.value for s in scenario_scores.scenarios[scenario]])
            offset_scores = numpy_scores[1:]
            differences = numpy_scores[:-1] - offset_scores
            plt.plot(training_sizes[1:], differences, label=f"Scenario {scenario.value} Score Change")

    numpy_scores = np.array([s.value for s in scenario_scores.combined])
    offset_scores = numpy_scores[1:]
    differences = numpy_scores[:-1] - offset_scores
    plt.plot(training_sizes[1:], differences, label=f"All Scenarios Combined Score Change")            
    plt.title(f"{metric} Score Change Diagnostics on {dataset} {category.value} data".title())
    plt.xlabel("Trained Data Points")
    plt.ylabel(f"{metric} score".title())
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(selection_path / 'change.svg', bbox_inches='tight')
    plt.close('all')


    plt.plot(training_sizes[1:], differences, label=f"All Scenarios Combined Score Change")            
    plt.title(f"{metric} Score Change Diagnostics on {dataset} {category.value} data".title())
    plt.xlabel("Trained Data Points")
    plt.ylabel(f"{metric} score".title())
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(selection_path / 'change_only_combined.svg', bbox_inches='tight')
    plt.close('all')

def visualise_model_diagnostics(record: ml.TrainingRecord, category: ml.Category,  save_dir: Path, metric: str, dataset: str):
    scenario_scores = record.scores[category].dataset_scores[metric].scores[dataset]
    selection_path = save_dir / category.value / dataset / metric / 'diagnostics'
    selection_path.mkdir(exist_ok=True, parents=True)
    visualise_std_diagnostic(scenario_scores=scenario_scores, training_sizes=record.training_sizes, selection_path=selection_path, metric=metric, dataset=dataset, category=category)
    visualise_interval_diagnostic(scenario_scores=scenario_scores, training_sizes=record.training_sizes, selection_path=selection_path, metric=metric, dataset=dataset, category=category)
    visualise_progression_diagnostics(scenario_scores=scenario_scores, training_sizes=record.training_sizes, selection_path=selection_path, metric=metric, dataset=dataset, category=category)
    for slope_length in (5, 10, 20):
        visualise_slope_diagnostics(scenario_scores=scenario_scores, training_sizes=record.training_sizes, selection_path=selection_path, metric=metric, dataset=dataset, category=category, slope_length=slope_length)


    

def visualise_metric_dataset(record: ml.TrainingRecord, category: ml.Category,  save_dir: Path, metric: str, dataset: str):
    scoring = record.scores[category].dataset_scores
    scenario_scores = scoring[metric].scores[dataset]
    selection_path = save_dir / category.value / dataset / metric / 'score'
    selection_path.mkdir(exist_ok=True, parents=True)
    maxima = []
    for scenario in Scenario:
        if scenario_scores.scenarios[scenario]:
            scores, minimum, maximum = zip(*[(score.value, score.minimum, score.maximum) for score in scenario_scores.scenarios[scenario]])
            maxima.extend(maximum)
            scores = np.array(scores)
            plt.plot(record.training_sizes, scores, label=f"Scenario {scenario.value}")
            plt.fill_between(record.training_sizes, minimum, maximum, alpha=0.2)
    scores, minimum, maximum = zip(*[(score.value, score.minimum, score.maximum) for score in scenario_scores.combined])
    maxima.extend(maximum)
    
    
    maxima = [m for m in maxima if not np.isnan(m)]
    if maxima:
        maxima.sort()
        percentile80 = maxima[int(len(maxima) * 0.8)]
        scores = np.array(scores)
        plt.ylim(top=percentile80)
        if plt.ylim()[0] < 0:
            plt.ylim(bottom=0)
        if percentile80 < 1:
            plt.ylim(top=1)
        else:
            plt.ylim(top=percentile80)
    plt.ylim(bottom=0)
    if plt.ylim()[1] < 1:
        plt.ylim(top=1)
    plt.title(f"{metric} Score on {dataset} {category.value} data".title())
    plt.xlabel("Trained Data Points")
    plt.ylabel(f"{metric} score".title())

    plt.plot(record.training_sizes, scores, label=f"All Scenarios Combined")
    plt.fill_between(record.training_sizes, minimum, maximum, alpha=0.2)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(selection_path / 'all.svg', bbox_inches='tight')
    plt.close('all')

    if maxima:
        maxima.sort()
        percentile80 = maxima[int(len(maxima) * 0.8)]
        scores = np.array(scores)
        plt.ylim(top=percentile80)
        if plt.ylim()[0] < 0:
            plt.ylim(bottom=0)
        if percentile80 < 1:
            plt.ylim(top=1)
        else:
            plt.ylim(top=percentile80)
    plt.ylim(bottom=0)
    if plt.ylim()[1] < 1:
        plt.ylim(top=1)
    plt.title(f"{metric} Score on {dataset} {category.value} data".title())
    plt.xlabel("Trained Data Points")
    plt.ylabel(f"{metric} score".title())

    plt.plot(record.training_sizes, scores, label=f"All Scenarios Combined")
    plt.fill_between(record.training_sizes, minimum, maximum, alpha=0.2)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(selection_path / 'only_combined.svg', bbox_inches='tight')
    plt.close('all')


    

if __name__ == "__main__":
    main()


