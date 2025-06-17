from custom_lib import data, ml, stats, vis
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

from matplotlib import pyplot as plt

rng = np.random.default_rng()

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.utils._testing import ignore_warnings


from modAL.models import ActiveLearner
from modAL.batch import uncertainty_batch_sampling
from modAL.models import CommitteeRegressor
from modAL.disagreement import max_std_sampling



train_data = data.load_data(Path(__file__).parent.parent /'combined-scan'/'samples.csv')
data.all_augments(train_data)
data.feature_engineer(train_data)
train_data = data.remove_low_filter(train_data)
bootstrap_data = train_data.sample(n=100)
features, labels = ml.split_into_data_and_label(train_data)


test_data = Path(__file__).parent.parent /'ppdb'/'inferred.csv'
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




single_pool_size = 15_000
total_points = 10000
batch_size_min = 2_0
batch_size_max = 2_1#int(total_points / 10)
bootstrap_size = int(total_points / 5)
number_of_learners = 1

@ignore_warnings()
def setup_learner():
    bootstrap_index = rng.choice(features.shape[0], bootstrap_size, replace=False)
    bootstrap_features = features[bootstrap_index]
    bootstrap_labels = labels[bootstrap_index]
    learner = ActiveLearner(
        estimator = clone(model),
        query_strategy=ml.GP_regression_std, 
        X_training=bootstrap_features,
        y_training=bootstrap_labels
    )
    return learner

custom_metric = stats.make_custom_metric(greater_is_better = False)
false_negative_metric = stats.make_false_negative_metric(greater_is_better = False)
false_positive_metric = stats.make_false_positive_metric(greater_is_better = False)

@ignore_warnings()
def train_learner(learner, batchsize, test_features, test_labels):
    result = ml.TrainingRecord(
        model = learner,
        batchsize = batchsize
    )
    pool_index = np.random.choice(features.shape[0], single_pool_size, replace=False)
    feature_pool = features[pool_index]
    label_pool = labels[pool_index]
    if total_points > feature_pool.shape[0]:
        raise ValueError(f"Total points ({total_points=}) larger than available samples ({feature_pool.shape[0]})")
    score_time = timedelta(0)
    start_time = datetime.now()
    metrics = {"custom": custom_metric, "false_positive": false_positive_metric, 
               "false_negative": false_negative_metric, "r2": r2_score}
    for name in metrics.keys():
        result.scores[name] = []
    for i in range(bootstrap_size, total_points, batchsize):
        query_idx, _ = learner.query(feature_pool, n_instances=min(batchsize, total_points - i))
        # find values for indexes
        learner.teach(feature_pool[query_idx], label_pool[query_idx])
        feature_pool = np.delete(feature_pool, query_idx, axis=0)
        label_pool = np.delete(label_pool, query_idx, axis=0)
        end_teach = datetime.now()
        result.training_sizes.append(i)#learner.X_training.shape[0])
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
    
    batchsize = rng.integers(batch_size_min, batch_size_max)    
    learner = setup_learner()
    return train_learner(learner, batchsize, test_features, test_labels)

records = []
for index in range(number_of_learners):
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