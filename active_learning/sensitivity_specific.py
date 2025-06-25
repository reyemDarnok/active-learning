from abc import abstractmethod, abstractproperty
from argparse import ArgumentParser, Namespace
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Generic, Iterable, List, Sequence, Set, Tuple, TypeVar

from matplotlib import pyplot as plt
plt.switch_backend('agg')
import pandas

from sys import path
path.append(str(Path(__file__).parent.parent))

from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.ioTypes.gap import Scenario
from focusStepsPelmo.ioTypes.pelmo import PelmoResult
from focusStepsPelmo.pelmo.local import run_local
from focusStepsPelmo.util.conversions import EnhancedJSONEncoder
from focusStepsPelmo.util.datastructures import correct_type

RealNumber = TypeVar("RealNumber", int, float)
 
def main():
    args = parse_args()
    if args.periods % 2 == 0:
        args.periods += 1
    base_sample: Combination = next(Combination.from_path(args.base_sample))
    fuzz_instruction: Dict[str, Dict[str, List[float]]] = {
        'compound.dt50.soil': {
            'compound.koc': [*range(10,100,10), *range(100,1000,50)]
        },
        'compound.koc': {
            'compound.dt50.soil': [*range(1,10), *range(10,50, 5), *range(50,200,10)]
        }
    }
    fuzz_sets: Dict[str, List[Combination]] = {}
    for to_fuzz, sample_mods in fuzz_instruction.items():
        fuzz_sets[to_fuzz] = []
        for mod_property, mod_values in sample_mods.items():
            particles = mod_property.split('.')
            for mod_value in mod_values:
                local_base: Dict[str, Any] = base_sample.asdict()
                last_dict = local_base
                for particle in particles[:-1]:
                    last_dict = last_dict[particle]
                last_dict[particles[-1]] = mod_value
                fuzz_sets[to_fuzz].append(Combination(**local_base))

    fuzzed_sets = {to_fuzz: list(fuzz_combinations(
        combinations=samples, property_name=to_fuzz, 
        maximum_reduction=args.maximum_reduction, maximum_increase=args.maximum_increase,
        periods=args.periods
    )) for to_fuzz, samples in fuzz_sets.items()}

    combinations_root: Path = args.work_dir / 'combinations'
    pelmo_work_root: Path = args.work_dir / 'pelmo'
    pelmo_output_root: Path = args.output_dir / 'pelmo'
    graph_output_root: Path = args.output_dir / 'graphs'
    graph_output_root.mkdir(parents=True, exist_ok=True)
    for property_name, fuzzed_set in fuzzed_sets.items():
        property_subpath = Path(property_name)
        (combinations_root / property_subpath).mkdir(exist_ok=True, parents=True)
        (pelmo_work_root / property_subpath).mkdir(exist_ok=True, parents=True)
        (pelmo_output_root / property_subpath).mkdir(exist_ok=True, parents=True)
        results: Dict[float, pandas.DataFrame] = {}
        for factor, combinations in fuzzed_set:
            combinations_path = combinations_root / property_subpath / f"{factor}.json"
            combinations_path.write_text(json.dumps(combinations, cls=EnhancedJSONEncoder))
            output_path = pelmo_output_root / property_subpath / f"{factor}.csv"
            run_local(work_dir=pelmo_work_root / property_subpath / str(factor),
                      output_file=output_path ,
                      combination_dir=combinations_path,
                      scenarios=args.scenarios)
            results[factor] = (pandas.read_csv(output_path))
        abs_results_frame = results[0]
        rel_results_frame = abs_results_frame.copy()
        abs_results_frame['Base PEC'] = abs_results_frame['0.compound_pec']
        abs_props = ['Base PEC']
        rel_results_frame['Unchanged'] = 1
        rel_props = ['Unchanged']
        for factor, result in results.items():
            if factor != 0:
                if factor > 0:
                    column_name = f'Increased by {factor*100:3.2f}%'
                else:
                    column_name = f"Decreased by {-factor*100:3.2f}%"
                abs_results_frame[column_name] = result['0.compound_pec']
                abs_props.append(column_name)
                rel_results_frame[column_name] = result['0.compound_pec'] / abs_results_frame['Base PEC']
                rel_props.append(column_name)
        changing_property = list(fuzz_instruction[property_name].keys())[0]
        pretty_changing = changing_property.split('.')[1].title()
        pretty_mod = property_name.split('.')[1].title()
        abs_results_frame.plot(x=f"combination.{changing_property}", y=abs_props, title=f"Changing {pretty_mod} with different {pretty_changing}")
        plt.ylabel('PEC')
        plt.xlabel(pretty_changing)
        plt.savefig(graph_output_root / f"{property_subpath}.svg", bbox_inches='tight')
        plt.close('all')

        rel_results_frame.plot(x=f"combination.{changing_property}", y=rel_props, title=f"Ratios after Fuzzing {pretty_mod} over different {pretty_changing}")
        plt.ylabel('Ratio Modified PEC/Base PEC')
        plt.xlabel(pretty_changing)
        plt.savefig(graph_output_root / f"{property_subpath}_ratios.svg", bbox_inches='tight')
        plt.close('all')
        
        

def fuzz_combinations(combinations: Sequence[Combination], property_name: str,
                        maximum_reduction: float = 0.1, maximum_increase: float = 0.1, 
                        periods: int = 10) -> Generator[Tuple[float, List[Combination]], None, None]:
        particles = property_name.split('.')
        still_negative = True
        for offset in range(periods):
            factor = (-maximum_reduction) + (maximum_reduction + maximum_increase) / (periods - 1) * offset
            if still_negative and factor > 0:
                yield 0, make_single_fuzz(factor=0, combinations=combinations, particles=particles)
                still_negative = False
            if factor != 0:
                yield factor, make_single_fuzz(factor=factor, combinations=combinations, particles=particles)
        if still_negative:
            yield 0, make_single_fuzz(factor=0, combinations=combinations, particles=particles)
   
def make_single_fuzz(factor: float, combinations: Sequence[Combination], particles: List[str]) -> List[Combination]:
    new_combinations: List[Combination] = []
    for combination in combinations:
        local_base = combination.asdict()
        last_dict = local_base
        for particle in particles[:-1]:
            last_dict = last_dict[particle] # type: ignore
        last_dict[particles[-1]] = last_dict[particles[-1]] * (1+factor) # type: ignore
        new_combinations.append(Combination(**local_base)) # type: ignore
    return new_combinations
            

def float_greater_than(compare_val: float) -> Callable[[str], float]:
    def transformer(source: str) -> float:
        result = float(source)
        assert result > compare_val, f"Value {result} is not bigger than {compare_val} as would be required"
        return result
    return transformer

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-b' '--base-sample', dest='base_sample',required=True, type=Path, help="The directory containing the samples this test will fuzz")
    parser.add_argument('-w', '--work-dir', type=Path, default=Path('sensitivity_test_work_dir'),
                        help="The directory to use as scratch space for this process")
    parser.add_argument('-o', '--output-dir', type=Path, default=Path('sensitivity_test'), help="The directory where the results are output")
    parser.add_argument('-r', '--maximum-reduction', type=float_greater_than(0), default=0.1, help="The maximum percentage a given value will be reduced by")
    parser.add_argument('-i', '--maximum-increase', type=float_greater_than(0), default=0.1, help="The maximum percentage a given value will be increased by")
    parser.add_argument('-p', '--periods', type=int, default=10, help="how many steps between maximum reduction and increase to test")
    parser.add_argument('-s', '--scenarios', nargs='*', type=lambda x: correct_type(x, Scenario), default=frozenset(Scenario), help="Which scenario to consider")
    args = parser.parse_args()
    args.scenarios = frozenset(args.scenarios)
    return args

if __name__ == "__main__":
    main()