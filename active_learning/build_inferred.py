from dataclasses import replace
import itertools
import json
from pathlib import Path

from random import randint
from sys import path
path.append(str(Path(__file__).parent.parent))
from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.ioTypes.compound import Compound
from focusStepsPelmo.ioTypes.gap import GAP, RelativeGAP
from focusStepsPelmo.ioTypes.scenario import Scenario
from focusStepsPelmo.util.conversions import EnhancedJSONEncoder


path_to_inferred = Path(__file__).parent.parent / 'ppdb_source' / 'inferred2pelmo.json'
compounds = Compound.from_path(path=path_to_inferred)
print('Setup Compound Reader')

path_to_gap = Path(__file__).parent.parent / 'examples' / 'gap.json'
gap: RelativeGAP = next(GAP.from_path(path=path_to_gap)) # type: ignore
print('Read GAP')
gap_variations = [gap]#[replace(gap, bbch=new_bbch) for new_bbch in range(1,90,10)]

combinations = []
for compound in compounds:
    for single_gap in gap_variations:
        for scenario in Scenario:
            mod_gap = replace(gap, bbch=randint(1,90))
            try:
                combinations.append(Combination(compound=compound, gap=single_gap, scenarios=frozenset([scenario])))
            except ValueError:
                pass

out_path = Path(__file__).parent.parent / 'ppdb_combinations_with_stages.json'
with open(out_path, 'w') as out:
    json.dump(obj=combinations, fp=out, cls=EnhancedJSONEncoder)