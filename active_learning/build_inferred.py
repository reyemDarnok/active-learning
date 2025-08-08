from dataclasses import replace
import itertools
import json
from pathlib import Path

from random import choice, randint
from sys import path
path.append(str(Path(__file__).parent.parent))
from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.ioTypes.compound import Compound
from focusStepsPelmo.ioTypes.gap import GAP, FOCUSCrop, RelativeGAP
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
created_combinations = 0
for _ in range(10):
    for compound in compounds:
        for single_gap in gap_variations:
            for scenario in Scenario:
                found_combo = False
                while not found_combo:
                    mod_gap = replace(gap,
                                    modelCrop=choice(list(FOCUSCrop)),
                                    number_of_applications=randint(1,3),
                                    interval=randint(7,21),
                                    bbch=randint(1,90),
                                    rate=randint(100,b=1000),
                                    apply_every_n_years=randint(1,3),
                                    )
                    try:
                        candidate = Combination(compound=compound, gap=mod_gap, scenarios=frozenset([scenario]))
                        list(candidate.gap.application_data(next(candidate.scenarios.__iter__())))
                        
                    except (ValueError, IndexError):
                        pass
                    else:
                        combinations.append(Combination(compound=compound, gap=mod_gap, scenarios=frozenset([scenario])))
                        found_combo = True
                        created_combinations+=1
                        if created_combinations % 100 == 0:
                            print(created_combinations)


out_path = Path(__file__).parent.parent / 'ppdb_combinations_open_gap_var_crop.json'
if out_path.exists():
   all_combinations = json.loads(out_path.read_text())
else:
    all_combinations = combinations


with open(out_path, 'w') as out:
    json.dump(obj=all_combinations, fp=out, cls=EnhancedJSONEncoder)