import pandas
from typing import List, Tuple, Iterable
from pathlib import Path
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
import math
import numpy
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error


def pearson_p(x: Iterable[float], y: Iterable[float]) -> float:
    return pearsonr(x, y)[1] # type: ignore

def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    return pearsonr(x, y)[0] # type: ignore


def find_input_correlations(correlations: pandas.DataFrame):
    input_columns = correlations.columns[correlations.columns.str.endswith(".pec")]
    input_correlation = correlations.drop(input_columns, axis=0).drop(input_columns, axis=1)
    # Exclude 1 in diagonal from stats
    input_correlation.values[tuple([numpy.arange(input_correlation.shape[0])])*2] = numpy.nan
    return input_correlation
    
def calc_pec_correlations(correlation, correlations_p):
    pec_columns = correlation.columns[correlation.columns.str.endswith(".pec")]
    return correlation[pec_columns].drop(pec_columns, axis=0), correlations_p[pec_columns].drop(pec_columns, axis=0)

def show_pec_correlations(correlation, correlations_p):
    pec_correlations, pec_correlations_p = calc_pec_correlations(correlation, correlations_p)
    parent_correlations = pec_correlations
    parent_correlations['p-value'] = pec_correlations_p['parent.pec']
    print(parent_correlations)
    

def abs_rmse(y_true, y_predicted) -> float:
    y_true = numpy.power(10, y_true)
    y_predicted = numpy.power(10, y_predicted)
    y_predicted = numpy.minimum(y_predicted, 1e10)
    y_true = numpy.minimum(y_true, 1e10)
    return float(numpy.sqrt(mean_squared_error(y_true, y_predicted)))

def neg_abs_rmse(y_true, y_predicted):
    return -abs_rmse(y_true, y_predicted)

def root_mean_squared_error(y_true, y_predicted) -> float:
    return float(numpy.sqrt(mean_squared_error(y_true, y_predicted)))

def neg_root_mean_squared_error(y_true, y_predicted):
    return -root_mean_squared_error(y_true, y_predicted)

def wrong_bin_score(y_true, y_predicted, border: float = -1):
    return sum((y_true > border) ^ (y_predicted > border)) / y_true.size

def neg_wrong_bin_score(y_true, y_predicted, border: float = -1):
    return - wrong_bin_score(y_true, y_predicted, border)

def make_false_negative_metric(split: float = -1, greater_is_better = True):
    def metric(y_true, y_pred, *, greater_is_better = greater_is_better):
        if len(y_true.shape) > 1:
            y_true = numpy.ravel(y_true)
            y_pred = numpy.ravel(y_pred)
        should_be_under = y_pred[y_true < split]
        if should_be_under.shape[0] > 0:
            score =  sum(should_be_under > split) / should_be_under.shape[0]
        else:
            score = 0
        if greater_is_better:
            score *= -1
        return score
    return metric

def make_false_positive_metric(split: float = -1, greater_is_better = True):
    def metric(y_true, y_pred, *, greater_is_better = greater_is_better):
        if len(y_true.shape) > 1:
            y_true = numpy.ravel(y_true)
            y_pred = numpy.ravel(y_pred)
        should_be_over = y_pred[y_true > split]
        if should_be_over.shape[0] > 0:
            score = sum(should_be_over < split) / should_be_over.shape[0]
        else:
            score = 0
        if greater_is_better:
            score *= -1
        return score
    return metric

def make_long_jump_metric(lower_bound: float = -1.5, upper_bound: float = -0.5, greater_is_better = True):
    def metric(y_true, y_pred, *, greater_is_better = greater_is_better):
        if len(y_true.shape) > 1:
            y_true = numpy.ravel(y_true)
            y_pred = numpy.ravel(y_pred)
        true_value_in_lower_indexes = y_true < lower_bound
        true_value_in_higher_indexes = y_true > upper_bound
        true_value_in_middle = (y_true > lower_bound) & (y_true < upper_bound)
        jumps_upward = y_pred[true_value_in_lower_indexes] > upper_bound
        jumps_downward = y_pred[true_value_in_higher_indexes] < lower_bound
        jumps_from_middle = numpy.abs(y_pred[true_value_in_middle] - y_true[true_value_in_middle]) > (upper_bound - lower_bound / 2) & ((y_pred[true_value_in_middle] > upper_bound) | (y_pred[true_value_in_middle] < lower_bound))
        score = (jumps_downward.sum() + jumps_upward.sum() + jumps_from_middle.sum()) / y_true.shape[0]
        
        if greater_is_better:
            score *= -1
        return score
    return metric

def make_up_jump_metric(lower_bound: float = -1.5, upper_bound: float = -0.5, greater_is_better = True):
    def metric(y_true, y_pred, *, greater_is_better = greater_is_better):
        if len(y_true.shape) > 1:
            y_true = numpy.ravel(y_true)
            y_pred = numpy.ravel(y_pred)
        true_value_in_lower_indexes = y_true < lower_bound
        jumps_upward = y_pred[true_value_in_lower_indexes] > upper_bound
        score = jumps_upward.sum() / jumps_upward.shape[0]
        
        if greater_is_better:
            score *= -1
        return score
    return metric

def make_down_jump_metric(lower_bound: float = -1.5, upper_bound: float = -0.5, greater_is_better = True):
    def metric(y_true, y_pred, *, greater_is_better = greater_is_better):
        if len(y_true.shape) > 1:
            y_true = numpy.ravel(y_true)
            y_pred = numpy.ravel(y_pred)
        true_value_in_higher_indexes = y_true > upper_bound
        jumps_downward = y_pred[true_value_in_higher_indexes] < lower_bound
        score = jumps_downward.sum() / jumps_downward.shape[0]
        
        if greater_is_better:
            score *= -1
        return score
    return metric

class CatboostCustom:

    def __init__(self, *args, **kwargs):
        self.metric = make_custom_metric(*args, **kwargs)

    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False
    
    def evaluate(self, approxes, target, weight):
        self.metric(target, approxes, input_weights=weight)

def make_custom_metric(
        max_weight: float = 10, 
        min_weight: float = 1,
        penalty_weight: float = 10, 
        falloff: float = 2,
        center: float = -1,
        greater_is_better: bool = True):
    def metric(y_true, y_pred, *, greater_is_better = greater_is_better, input_weights = None):
        total_weight = 0
        total_score = 0
        if len(y_true.shape) > 1:
            y_true = numpy.ravel(y_true)
            y_pred = numpy.ravel(y_pred)
        if input_weights is None:
            input_weights = numpy.ones_like(y_true)
        for y_t, y_p, i_w in zip(y_true, y_pred, input_weights):
            if (y_t < center < y_p) or (y_p < center < y_t):
                weight = penalty_weight * i_w
            else:
                weight = max(max_weight - falloff * abs(y_t - center), min_weight) * i_w
            diff = abs((y_t - y_p) / y_t)
            total_score += weight * diff
            total_weight += weight
        score = total_score / total_weight
        if greater_is_better:
            score *= -1
        return score
            
    return metric

def make_custom_rmse_metric(
        max_weight: float = 10, 
        min_weight: float = 1,
        penalty_weight: float = 10, 
        falloff: float = 2,
        center: float = -1,
        greater_is_better: bool = True):
    def metric(y_true, y_pred, *, greater_is_better = greater_is_better, input_weights = None):
        total_weight = 0
        total_score = 0
        if len(y_true.shape) > 1:
            y_true = numpy.ravel(y_true)
            y_pred = numpy.ravel(y_pred)
        if input_weights is None:
            input_weights = numpy.ones_like(y_true)
        for y_t, y_p, i_w in zip(y_true, y_pred, input_weights):
            if (y_t < center < y_p) or (y_p < center < y_t):
                weight = penalty_weight * i_w
            else:
                weight = max(max_weight - falloff * abs(y_t - center), min_weight) * i_w
            diff = (y_t - y_p) ** 2
            total_score += weight * diff
            total_weight += weight
        score = total_score / total_weight
        if greater_is_better:
            score *= -1
        return score
            
    return metric


def pec_std_metric(committee_pred: List[NDArray],  *, greater_is_better = True, input_weights = None):
    if input_weights is None:
        input_weights = numpy.ones_like(committee_pred[0].shape)
    acc = 0
    for input_weight, *predictions in zip(input_weights, *committee_pred):
        acc += numpy.std(predictions) * input_weight
    score = acc / numpy.sum(input_weights)
    if greater_is_better:
        score *= -1
    return score

def pec_interval_metric(committee_pred: List[NDArray],  *, greater_is_better = True, input_weights = None):
    if input_weights is None:
        input_weights = numpy.ones_like(committee_pred[0].shape)
    acc = 0
    for input_weight, *predictions in zip(input_weights, *committee_pred):
        acc += (numpy.max(predictions) - numpy.min(predictions)) * input_weight
    score = acc / numpy.sum(input_weights)
    if greater_is_better:
        score *= -1
    return score



def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    Source:
    ------
        https://stackoverflow.com/a/11886564
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = numpy.median(points, axis=0)
    diff = numpy.sum((points - median)**2, axis=-1)
    diff = numpy.sqrt(diff)
    med_abs_deviation = numpy.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh