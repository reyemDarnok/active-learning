import json
from pathlib import Path

from sys import path
path.append(str(Path(__file__).parent.parent))
from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.ioTypes.compound import Compound
from focusStepsPelmo.ioTypes.gap import GAP
from focusStepsPelmo.ioTypes.scenario import Scenario
from focusStepsPelmo.util.conversions import EnhancedJSONEncoder


path_to_inferred = Path(__file__).parent.parent / 'ppdb_source' / 'inferred2pelmo.json'
compounds = Compound.from_path(path=path_to_inferred)
print('Setup Compound Reader')

path_to_gap = Path(__file__).parent.parent / 'examples' / 'gap.json'
gap = next(GAP.from_path(path=path_to_gap))
print('Read GAP')

combinations = []
i = 0
for compound in compounds:
    i += 1
    s_i = 0
    for scenario in Scenario:
        try:
            combinations.append(Combination(compound=compound, gap=gap, scenarios=frozenset([scenario])))
            s_i+=1
        except ValueError:
            pass
    print(f"Wrote Combination number {i:5} with {s_i} scenarios")

out_path = Path(__file__).parent.parent / 'ppdb_combinations.json'
with open(out_path, 'w') as out:
    json.dump(obj=combinations, fp=out, cls=EnhancedJSONEncoder)