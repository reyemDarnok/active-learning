from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas
from matplotlib import pyplot as plt
import math
import numpy

def limit_column(column: pandas.Series, whisker_size: float = 4) -> Tuple[float, float]:
    lower = max(column.min(), column.quantile(0.25) - (column.quantile(0.5) - column.quantile(0.25)) * whisker_size)
    upper = min(column.max(), column.quantile(0.75) + (column.quantile(0.75) - column.quantile(0.5)) * whisker_size)
    return lower, upper

def describe(column_name: str, to_describe: pandas.DataFrame, save_dir, save_to, range: Optional[Tuple[float, float]]=None, whisker_size: float = 4,
            ):
    column = to_describe[column_name]
    #fontsize = 28
    if range is None:
        range = limit_column(column, whisker_size)
    print(column_name)
    print('Unmodified')
    print(column.describe(datetime_is_numeric=True)) # type: ignore
    axes = column.plot(use_index=True, kind='hist', bins=40, range=range)
    axes.set_title(column_name)
    #axes.tick_params(axis='both', which='major', labelsize=fontsize)
    #axes.tick_params(axis='both', which='minor', labelsize=fontsize // 0.75)
    
    (save_dir / save_to).mkdir(exist_ok=True, parents=True)
    plt.savefig(save_dir / save_to / 'normal.svg', bbox_inches='tight')
    plt.show()
    print('Log scale')
    try:
        no_zero = column[column != 0]
        log_column = no_zero.apply(math.log10)
        print(log_column.describe(datetime_is_numeric=True)) # type: ignore
        log_column = no_zero.apply(math.log10)
        axes = log_column.plot(use_index=True, kind='hist', bins=40, range=limit_column(log_column, whisker_size))
        axes.set_title(f"{column_name} Log10")
        #axes.tick_params(axis='both', which='major', labelsize=fontsize)
        #axes.tick_params(axis='both', which='minor', labelsize=fontsize // 0.75)
        plt.savefig(save_dir / save_to / 'log.svg', bbox_inches='tight')
        plt.show()
    except ValueError:
        print('Error during log application')
    except TypeError:
        print('Error during log application')
    print()
    
def score_predictions(predictions, metric, lims=None):
    scores = {}
    true_prediction = predictions['pelmo'].values
    for name in predictions.columns:
        model_prediction = predictions[name]
        scores[name] = metric(true_prediction, model_prediction)
    return scores


def show_cv_scores(scores):
    name_length = max(*(len(x) for x in scores.keys())) + 1
    display_scores = scores.copy()
    for name, score in display_scores.items():
        print(f'{name:<{name_length}}: {score:.2f}')
        

def plot_model(label_data, prediction, name, lims=None, save_to=None):
    print(name)
    #fontsize = 20
    markersize = 10
    figure, axes = plt.subplots(ncols=1)
    ylim = label_data.min(), label_data.max()
    if lims != None:
        ylim = max(lims[0], ylim[0]), min(lims[1], ylim[1])
    xlim = ylim
    axes.scatter(x=prediction, y=label_data, s=markersize)
    axes.set_xlabel(f'{name} Predicted PEC log10')#, fontsize = fontsize)
    #axes.tick_params(axis='both', which='major', labelsize = fontsize)
    #axes.tick_params(axis='both', which='minor', labelsize = fontsize // 0.75)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_ylabel(f'Actual PEC log10')#, fontsize = fontsize)
    axes.axline((-1,-1), slope=1, color='red')
    if save_to is not None:
        save_to.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_to / f'{name}.svg', bbox_inches='tight')
    plt.show()
    def exp10(x):
        return 10 ** min(20,x)
    figure, axes = plt.subplots(ncols=1)
    ylim = exp10(ylim[0]), exp10(ylim[1])
    xlim = ylim
    axes.scatter(x=[exp10(x) for x in prediction], y=[exp10(x) for x in label_data], s=markersize)
    axes.set_xlabel(f'{name} Predicted PEC')#, fontsize = fontsize)
    #axes.tick_params(axis='both', which='major', labelsize = fontsize)
    #axes.tick_params(axis='both', which='minor', labelsize = fontsize // 0.75)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_ylabel(f'Actual PEC')#, fontsize = fontsize)
    axes.axline((-1,-1), slope=1, color='red')
    if save_to is not None:
        save_to.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_to / f'{name}-abs.svg', bbox_inches='tight')
    plt.show()
    

def plot_predictions(predictions, *, save_root, save_dir,  lims=None):
    label = predictions['pelmo']
    for name in predictions.columns:
        prediction = predictions[name]
        plot_model(label, prediction, name, lims=lims, save_to = save_root / save_dir)

           
def score_and_show(predictions, metric, lims=None):
    show_cv_scores(score_predictions(predictions, metric, lims=lims))
    
def make_box_plot(scores, save_dir, save_file):
    plt.boxplot([x for x in scores.values()], labels=[x for x in scores.keys()]) # type: ignore
    #fontsize = 28
    axes = plt.gca()
    #axes.tick_params(axis='both', which='major', labelsize = fontsize)
    #axes.tick_params(axis='both', which='minor', labelsize = fontsize // 0.75)
    plt.ylim(0,1)
    plt.xticks(rotation=90)
    save_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_dir / save_file, bbox_inches='tight')
    plt.show()
    
def score_bars(predictions_dict, metric, title, ylims=(0,1), save_to: Optional[Path] = None):
    number_of_predictions = len(predictions_dict.keys())
    sample_predictions = list(predictions_dict.values)[0]
    x_labels = sample_predictions.keys()
    base_index = numpy.arange(len(sample_predictions.keys()))
    total_width_of_group = 0.8
    fill_ratio = 0.8
    width = total_width_of_group/number_of_predictions
    bar_groups = {}
    for offset, t in enumerate(predictions_dict.items()):
        dataset_name, predictions = t
        offset -= number_of_predictions / 2 - 0.5# enumerate can't use float offsets
        scores = score_predictions(predictions, metric)
        bar_groups[dataset_name] = plt.bar(base_index + width * offset, [metric(predictions['pelmo'], predictions[name]) for name in predictions.columns], width*fill_ratio)
    plt.ylim(ylims)
    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=90)
    plt.legend( [bar[0] for bar in bar_groups.values()], bar_groups.keys(), loc="lower left", bbox_to_anchor=(1.04, 0))
    if title is not None:
        plt.title(title)
    if save_to is not None:
        save_to.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_to / f"{title}.svg", bbox_inches='tight')
    plt.show()
    
def cross_val(cross_val_results, save_to =None):
    number_of_datasets = len(cross_val_results.keys())
    sample_result = list(cross_val_results.values())[0]
    sample_model_result = list(sample_result.values())[0]
    x_labels = sample_result.keys()
    base_index = numpy.arange(len(sample_result.keys()))
    total_width_of_group = 0.8
    fill_ratio = 0.8
    width = total_width_of_group/number_of_datasets
    box_groups = {}
    for metric in sample_model_result.keys():
        for offset, t in enumerate(cross_val_results.items()):
            dataset_name, cross_val_result = t
            offset -= number_of_datasets / 2 - 0.5
            box_data = [scorecard[metric] for scorecard in cross_val_result.values()]
            box_groups[dataset_name] = plt.boxplot(box_data, positions=base_index + width * offset, widths=width * fill_ratio, patch_artist=True, showfliers=False)
        plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=90)
        plt.gca().set_ylim(bottom=0)
        if metric == "R²":
            plt.gca().set_ylim(top=1)
        colormap = plt.cm.get_cmap('plasma', number_of_datasets)
        for index, box_group in enumerate(box_groups.values()):
            group_color = colormap(index)
            for patch in box_group['boxes']:
                patch.set_color(group_color)
        plt.legend( [box["boxes"][0] for box in box_groups.values()], box_groups.keys(), loc="lower left", bbox_to_anchor=(1.04, 0))
        plt.title(metric)
        if save_to is not None:
            save_to.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_to / f"{metric}.svg", bbox_inches='tight')
        plt.show()