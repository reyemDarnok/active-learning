from matplotlib import pyplot as plt
import json
from pathlib import Path
import numpy as np

rec_file = Path('training', 'weighted_sampling_5', 'results', 'record.json')
data = json.loads(rec_file.read_text())


training_sizes = data['training_sizes']
rmse_score = data['scores']['External Test']['custom RMSE']['total']['combined']
rmse_val = [score['value'] for score in rmse_score]
rmse_minimum = [score['minimum'] for score in rmse_score]
rmse_maximum = [score['maximum'] for score in rmse_score]
plt.plot(training_sizes, rmse_val)
plt.fill_between(training_sizes, rmse_minimum, rmse_maximum, alpha=0.2)
rmse_maximum.sort()
percentile80 = rmse_maximum[int(len(rmse_maximum) * 0.8)]
rmse_val = np.array(rmse_val)
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
plt.xlabel("Trained Data Points")
plt.ylabel('Custom RMSE Score')
plt.title('Custom RMSE Score On Total External Test Data')
plt.savefig('Custom_RMSE.svg')