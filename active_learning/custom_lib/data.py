from typing import Optional
import pandas
from pathlib import Path
from datetime import datetime, timedelta
from sys import path
path.append(str(Path(__file__).parent.parent.parent))
from focusStepsPelmo.ioTypes.gap import FOCUSCrop, Scenario
import math


#Data initialisation

def load_data(file: Path) -> pandas.DataFrame:
    return pandas.read_csv(file, dtype={'scenario': 'category', 'gap.arguments.modelCrop': 'category'})

def date_parser(time: str) -> int:
    # pandas can't handle year 1 dates, so save as day of year
    return (datetime.fromisoformat(time) - datetime(year=1, month=1, day=1) + timedelta(days=1)).days

def transform_data_types(to_transform: pandas.DataFrame):
    for column in to_transform.columns:
        try:
            to_transform[column] = to_transform[column].astype('float64')
        except ValueError:
            pass
    
def rename_columns(to_transform: pandas.DataFrame):
    def rename(name: str) -> str:
        if name == '0.compound_pec':
            return 'parent.pec'
        elif name == '1.compound_pec':
            return 'metabolite.pec'
        elif name.startswith('combination.compound.metabolites'):
            return "metabolite." + name.split('.', maxsplit=3)[-1]
        elif name.startswith('combination.compound.'):
            return "parent." + name.split('.', maxsplit=2)[-1]
        elif name.startswith('combination.gap.'):
            return "gap." + name.split('.', maxsplit=2)[-1]
        elif name == 'combination.scenarios.0':
            return 'scenario'
        elif name.startswith('compound.metabolites'):
            return "metabolite." + name.split('.', maxsplit=2)[-1]
        elif name.startswith('compound.'):
            return "parent." + name.split('.', maxsplit=1)[-1]
        elif name.startswith('gap.'):
            return "gap." + name.split('.', maxsplit=1)[-1]
        elif name == 'combination.scenarios.0':
            return 'scenario'
        else:
            return name
            
    to_transform.rename(columns={x: rename(x) for x in to_transform.columns}, inplace=True)
    
def add_calculated(to_transform: pandas.DataFrame):
    if 'metabolite.molarMass' in to_transform:
        to_transform['metabolite.henry'] = (to_transform['metabolite.vapor_pressure'] * to_transform['metabolite.molarMass'] /
                                            to_transform['metabolite.water_solubility'])
        to_transform['metabolite.log_henry'] = to_transform['metabolite.henry'].apply(math.log10)
        to_transform['metabolite.formation_fraction_mass'] = (to_transform['metabolite.formation_fraction'] * 
                                                          to_transform['metabolite.molarMass'] / to_transform['parent.molarMass'])

    to_transform['parent.henry'] = (to_transform['parent.vapor_pressure'] * to_transform['parent.molarMass'] / 
                                    to_transform['parent.water_solubility'])
    to_transform['parent.log_henry'] = to_transform['parent.henry'].apply(lambda x: math.log10(max(1e-10,x)))

def drop_impossible(to_transform):
    drop_index = to_transform[to_transform["parent.log_henry"] == float('inf')]
    to_transform.drop(drop_index.index, inplace=True)
    
def drop_uninteresting_artefacts(to_transform: pandas.DataFrame):
    # Drop names
    to_transform.drop(columns=to_transform.filter(regex='name'), inplace=True)
    to_transform.drop(columns=['gap.type'], inplace=True)

def all_augments(to_transform: pandas.DataFrame):
    rename_columns(to_transform)
    transform_data_types(to_transform)
    add_calculated(to_transform)
    drop_impossible(to_transform)
    drop_uninteresting_artefacts(to_transform)
    
    
def minimal_filter(to_transform: pandas.DataFrame) -> pandas.DataFrame:
    return remove_low_filter(to_transform, 0)
        
def remove_low_filter(to_transform: pandas.DataFrame, lower_bound=1e-4) -> pandas.DataFrame:
    return between_filter(to_transform, lower_bound, float('inf'))
        
def between_filter(to_transform: pandas.DataFrame, lower_bound: float = 1e-2, upper_bound: float = 3.16227766) -> pandas.DataFrame:
                                                                                                    # 10^0.5
    if "metabolite.pec" in to_transform:
        return to_transform[(to_transform['parent.pec'].between(lower_bound, upper_bound, inclusive="neither")) & (to_transform['metabolite.pec'].between(lower_bound, upper_bound, inclusive="neither"))]
    else:
        return to_transform[to_transform['parent.pec'].between(lower_bound, upper_bound, inclusive="neither")]

def minimal_filter_raw(to_transform: pandas.DataFrame) -> pandas.DataFrame:
    return remove_low_filter_raw(to_transform, 0)
        
def remove_low_filter_raw(to_transform: pandas.DataFrame, lower_bound=1e-4) -> pandas.DataFrame:
    return between_filter_raw(to_transform, lower_bound, float('inf'))
        
def between_filter_raw(to_transform: pandas.DataFrame, lower_bound: float = 1e-2, upper_bound: float = 3.16227766) -> pandas.DataFrame:
                                                                                                    # 10^0.5
    if "1.compound_pec" in to_transform:
        return to_transform[(to_transform['0.compound_pec'].between(lower_bound, upper_bound, inclusive="neither")) & (to_transform['1.compound_pec'].between(lower_bound, upper_bound, inclusive="neither"))]
    else:
        return to_transform[to_transform['0.compound_pec'].between(lower_bound, upper_bound, inclusive="neither")]

def create_datasets(to_transform: pandas.DataFrame):
    datasets = {}
        
    round_up_df = to_transform.copy()
    round_up_df['parent.pec'] = round_up_df['parent.pec'].apply(lambda x: max(x, 1e-4))
    if "metabolite.pec" in round_up_df:
        round_up_df['metabolite.pec'] = round_up_df['metabolite.pec'].apply(lambda x: max(x, 1e-4))
    datasets['round_up'] = round_up_df

    as_is_df = to_transform.copy()
    if "metabolite.pec" in as_is_df:
        as_is_df = as_is_df[(as_is_df['parent.pec'] != 0) & (as_is_df['metabolite.pec']) != 0]
    else:
         as_is_df = as_is_df[(as_is_df['parent.pec'] != 0)]
    datasets['as_is'] = as_is_df

    drop_any_df = to_transform.copy()
    if "metabolite.pec" in drop_any_df:
        drop_any_df = drop_any_df[(drop_any_df['parent.pec'] > 1e-4) & (drop_any_df['metabolite.pec'] > 1e-4)]
    else:
        drop_any_df = drop_any_df[(drop_any_df['parent.pec'] > 1e-4)]
    datasets['drop_any'] = drop_any_df

    drop_all_df = to_transform.copy()
    if "metabolite.pec" in drop_all_df:
        drop_all_df = drop_all_df[((drop_all_df['parent.pec'] > 1e-4) & (drop_all_df['metabolite.pec'] > 0))
                              | ((drop_all_df['parent.pec'] > 0) & (drop_all_df['metabolite.pec'] > 1e-4))]
    else:
        drop_all_df = drop_all_df[(drop_all_df['parent.pec'] > 1e-4)]
    datasets['drop_all'] = drop_all_df
    return datasets

def reduce(dataset: pandas.DataFrame, lower_limit: Optional[float] = None, upper_limit: Optional[float] = None) -> pandas.DataFrame:
    reduced = dataset
    if lower_limit is not None:
        reduced = reduced[reduced['parent.pec'] > lower_limit]
    if upper_limit is not None:
        reduced = reduced[reduced['parent.pec'] < upper_limit]
    return reduced

def feature_engineer(to_transform: pandas.DataFrame):
    def cube(x: float) -> float:
        return x**3
    to_transform['parent.koc'] = to_transform['parent.koc'].apply(cube)
    to_transform['parent.dt50.soil'] = to_transform['parent.dt50.soil'].apply(cube)