"""Everything about machine learning in this project"""
from pathlib import Path

import pandas

"""Block load data"""
data_source = Path("scan-2024-07-26-copy/out/samples.csv")
df = pandas.read_csv(data_source, parse_dates=['gap.arguments.time_in_year'],
                     dtype={'scenario': 'category', 'gap.arguments.modelCrop': 'category'})