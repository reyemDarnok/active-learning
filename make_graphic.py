import math
from matplotlib import pyplot as plt
import pandas

df = pandas.read_csv('ppdb-scan/samples.csv')
renaming_rules = {'compound.dt50.soil': 'DT50 in log(days)',
                  'compound.freundlich': 'Freundlich Coefficient',
                  'compound.koc': 'Koc in log(L/Kg)',
                  'compound.water_solubility': 'Water Solubility in log(mg/L)',
                  'compound.vapor_pressure': 'Vapour Pressure in log(Pa)'}
df.rename(columns=renaming_rules, inplace=True)
for column in df.columns:
    if 'log(' in column:
        df = df[df[column] != 0]
        df[column] = df[column].apply(math.log10)
df = df[df['0.compound_pec'] > 1e-4]
df.hist(column=list(renaming_rules.values()))
plt.tight_layout()
plt.savefig('dset_dist.svg')
plt.close('all')
df.hist(column=list(renaming_rules.values()))
plt.tight_layout()
plt.savefig('dest_dist.png')