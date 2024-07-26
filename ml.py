"""Everything about machine learning in this project"""
from datetime import datetime
from pathlib import Path

import pandas

from focusStepsPelmo.ioTypes.gap import FOCUSCrop, Scenario

"""Block load data"""


def date_parser(time: str) -> datetime:
    return datetime.fromisoformat(time).replace(year=2001)


data_source = Path("scan-2024-07-26-copy/out/samples.csv")
working_df = pandas.read_csv(data_source, parse_dates=['gap.arguments.time_in_year'],
                             date_parser=date_parser,
                             dtype={'scenario': 'category', 'gap.arguments.modelCrop': 'category'})
loaded_df = working_df.copy()

"""Block transform data types"""


def transform_data_types(to_transform: pandas.DataFrame):
    # Drop names
    to_transform.drop(columns=to_transform.filter(regex='name'), inplace=True)
    to_transform.drop(columns=['gap.type'], inplace=True)
    to_transform['gap.arguments.modelCrop'] = to_transform['gap.arguments.modelCrop'].apply(
        lambda x: list(FOCUSCrop).index(FOCUSCrop.parse(x))
    )
    to_transform['scenario'] = to_transform['scenario'].apply(
        lambda x: list(Scenario).index(Scenario[x.split('.')[1]])
    )

transform_data_types(working_df)
transformed_df = working_df.copy()
