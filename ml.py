"""Everything about machine learning in this project"""
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas
from matplotlib import pyplot as plt
from pandas import DataFrame

from focusStepsPelmo.ioTypes.gap import FOCUSCrop, Scenario

"""Block load data"""

pandas.set_option('display.max_columns', None)
def date_parser(time: str) -> int:
    return (datetime.fromisoformat(time) - datetime(year=1, month=1, day=1) + timedelta(days=1)).days

def load_data(file: Path) -> DataFrame:
    return pandas.read_csv(data_source, dtype={'scenario': 'category', 'gap.arguments.modelCrop': 'category'})

data_source = Path("scan-2024-07-26-copy/out/samples.csv")
working_df = load_data(data_source)
loaded_df = working_df.copy()

"""Block transform data types"""


def transform_data_types(to_transform: pandas.DataFrame):
    # Drop names
    to_transform.drop(columns=to_transform.filter(regex='name'), inplace=True)
    to_transform.drop(columns=['gap.type'], inplace=True)
    to_transform['gap.arguments.time_in_year'] = to_transform['gap.arguments.time_in_year'].apply(date_parser)
    to_transform['gap.arguments.modelCrop'] = to_transform['gap.arguments.modelCrop'].apply(
        lambda x: list(FOCUSCrop).index(FOCUSCrop.parse(x))
    ).astype('int')
    to_transform['scenario'] = to_transform['scenario'].apply(
        lambda x: list(Scenario).index(Scenario[x.split('.')[1]])
    ).astype('int')

transform_data_types(working_df)
transformed_df = working_df.copy()

"""Block consistency validity"""
def describe(column_name: str, to_describe: DataFrame = working_df):
    print(column_name)
    print('Unmodified')
    print(to_describe[column_name].describe(datetime_is_numeric=True))
    to_describe[column_name].plot(use_index=True, kind='hist', bins=40)
    plt.show()
    print('Log scale')
    try:
        print(to_describe[column_name].apply(math.log).describe(datetime_is_numeric=True))
        to_describe[column_name].plot(use_index=True, kind='hist', bins=40, log=True)
    except ValueError:
        print('Error during log application')
    except TypeError:
        print('Error during log application')
    plt.show()
    print()

describe('compound.metabolites.0.formation_fraction')
describe('compound.metabolites.0.metabolite.dt50.soil')
describe('compound.metabolites.0.metabolite.freundlich')
describe('compound.metabolites.0.metabolite.koc')
describe('compound.metabolites.0.metabolite.molarMass')
describe('compound.metabolites.0.metabolite.plant_uptake')
describe('compound.metabolites.0.metabolite.vapor_pressure')
describe('compound.metabolites.0.metabolite.water_solubility')
describe('compound.dt50.soil')
describe('compound.freundlich')
describe('compound.koc')
describe('compound.molarMass')
describe('compound.plant_uptake')
describe('compound.water_solubility')
describe('gap.arguments.apply_every_n_years')
describe('gap.arguments.interval')
describe('gap.arguments.modelCrop')
describe('gap.arguments.number_of_applications')
describe('gap.arguments.rate')
describe('gap.arguments.time_in_year')
describe('gap.interception')
describe('scenario')
describe('0.compound_pec')
describe('1.compound_pec')

"""Compound has 0 plant_uptake, metabolite 0.5 - error in ppdb-scan.json but not damaging"""
"""PECs are mostly below 0.1 - dataset was taken from compounds already adapted, so high pecs are missing"""
failing_pecs = working_df.query('`0.compound_pec` > 0.1 or `1.compound_pec` > 0.1')
print(f"Number of PECs greater than 0.1 = {failing_pecs.shape[0]}")
print(f"That is {failing_pecs.shape[0] / working_df.shape[0] * 100} percent of all data points")

working_df.plot(use_index=True, kind='hist', y=['0.compound_pec', '1.compound_pec'], bins=20, range=(0,0.2))
plt.show()


correlations = working_df.corr()
for column in working_df.columns:
    if correlations[column].isna().all():
        correlations.drop(columns=[column], inplace=True)
        correlations.drop(column, inplace=True)
print(correlations[['0.compound_pec', '1.compound_pec']])
"""Greatest Correlation with dt50 at 0.26 for parent and 0.3 for metabolite
Lesser Correlations for freundlich (0.11 and 0.13) and koc (-0.11 and -0.13).
The largest Correlation from metabolite pec to parent properties is with parent water_solubility at 0.09
These are all weak correlations so while they might be useful for a first intuition, an actual model is still required
"""

correlations.filter()

