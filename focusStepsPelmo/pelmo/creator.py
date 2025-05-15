#!/usr/bin/env python3
"""A script for creating psm files"""
import asyncio
import json
import logging
from argparse import ArgumentParser, Namespace
from dataclasses import replace
from pathlib import Path
from typing import Generator, Iterable, Optional, Type, TypeVar, Union, Tuple, FrozenSet, List

from jinja2 import Environment, select_autoescape, StrictUndefined, PackageLoader

from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.ioTypes.compound import Compound
from focusStepsPelmo.ioTypes.gap import GAP, FOCUSCrop, Scenario
from focusStepsPelmo.pelmo.psm_file import PsmFile
from focusStepsPelmo.util import jsonLogger as jsonLogger

jinja_env = Environment(loader=PackageLoader('focusStepsPelmo.pelmo'),
                        autoescape=select_autoescape(), undefined=StrictUndefined)


def main():
    """Entrypoint for calling this script from the command line"""
    args = parse_args()
    compounds = None
    gaps = None
    combinations = None
    if args.compound_file and args.compound_file.is_dir():
        compounds = args.compound_file.rglob("*.json")
    elif args.compound_file:
        compounds = [args.compound_file]
    if args.gap_file and args.gap_file.is_dir():
        gaps = args.gap_file.rglob("*.json")
    elif args.gap_file:
        gaps = [args.gap_file]
    if args.combination_file and args.combination_file.is_dir():
        combinations = args.combination_file.rglob("*.json")
    elif args.combination_file:
        combinations = [args.combinations]
    write_psm_files(output_dir=args.output_dir, compounds=compounds, gaps=gaps, # type: ignore
                    combinations=combinations, pessimistic_interception=args.pessimistic_interception) # type: ignore


def write_psm_files(output_dir: Path, pessimistic_interception: bool,
                    compounds: Optional[Iterable[Union[Path, Compound]]] = None,
                    gaps: Optional[Iterable[Union[Path, GAP]]] = None,
                    combinations: Optional[Iterable[Union[Path, Combination]]] = None) -> int:
    """Writes psm files to the output_dir. psm files are named after the hash of their content, which as string hashes
    are not stable due to security considerations in python that do not matter for this script.
    Set the env var PYTHONHASHSEED for stable hashes.
    :param output_dir: Where to write the psm files to
    :param compounds: The compounds to combine with gaps to make psm files
    :param gaps: The gaps to combine with compounds to make psm files
    :param combinations: The combinations to turn into psm files
    :return: The number of psm files written"""
    if compounds:
        compounds = load_or_use(compounds, Compound)
    if gaps:
        gaps = load_or_use(gaps, GAP)
    if combinations:
        combinations = load_or_use(combinations, Combination)
    total = 0
    output_dir.mkdir(exist_ok=True, parents=True)
    for psm_file in generate_psm_files(compounds=compounds, gaps=gaps, combinations=combinations, pessimistic_interception=pessimistic_interception): # type: ignore
        total += 1
        (output_dir / f"{hash(psm_file)}.psm").write_text(psm_file[0], encoding="windows-1252")
    return total


T = TypeVar('T')


def load_or_use(it: Iterable[Union[Path, T]], t: Type[T]) -> Generator[T, None, None]:
    """Takes an iterable of type T or Paths to objects that can be parsed to T
    :param it: The mixed iterable
    :param t: The type to parse to
    :return: The type T values in it"""
    for element in it:
        if isinstance(element, t):
            yield element
        else:
            yield from t.from_path(element) # type: ignore


def generate_psm_files(compounds: Optional[Iterable[Compound]] = None, gaps: Optional[Iterable[GAP]] = None,
                       crops: FrozenSet[FOCUSCrop] = frozenset(FOCUSCrop),
                       scenarios: FrozenSet[Scenario] = frozenset(Scenario),
                       combinations: Optional[Iterable[Combination]] = None,
                       pessimistic_interception: bool = False) -> Generator[Tuple[str, FOCUSCrop, FrozenSet[Scenario]], None, None]:
    """Create the contents of psm files
    :param compounds: The compounds to combine with gaps to make psm files
    :param gaps: The gaps to combine with compounds to make psm files
    :param combinations: The combinations to turn into psm files
    :param crops: The crops for which psm files should be generated. If a GAP does not match the crops given, it will be
    skipped
    :param scenarios:The scenarios for which psm files should be generated. For a given GAP, the scenarios used are
    the intersection of the scenarios defined by the gap and the scenarios passed into the function
    :return: The contents of the psm files"""
    assert not (bool(compounds) ^ bool(gaps)), "Either both or neither of compound file have to be specified"
    if combinations:
        for combination in combinations:
            psm_file_scenarios = scenarios.intersection(combination.gap.defined_scenarios)
            comment = json.dumps({"combination": hash(combination)})
            yield _generate_psm_contents(compound=combination.compound, gap=combination.gap,
                                         comment=comment, pessimistic_interception=pessimistic_interception), combination.gap.modelCrop, psm_file_scenarios
    if compounds and gaps:
        compounds = list(compounds)
        for gap in gaps:
            if gap.modelCrop in crops:
                for compound in compounds:
                    psm_file_scenarios = scenarios.intersection(gap.defined_scenarios)
                    comment = json.dumps({"compound": hash(compound), "gap": hash(gap)})
                    yield _generate_psm_contents(compound, gap, comment, pessimistic_interception=pessimistic_interception), gap.modelCrop, psm_file_scenarios


async def generate_psm_files_async(compounds: Optional[Iterable[Compound]] = None, gaps: Optional[Iterable[GAP]] = None,
                                   crops: FrozenSet[FOCUSCrop] = frozenset(FOCUSCrop),
                                   scenarios: FrozenSet[Scenario] = frozenset(Scenario),
                                   combinations: Optional[Iterable[Combination]] = None,
                                   pessimistic_interception: bool = False
                                   ) -> List[Tuple[str, FOCUSCrop, FrozenSet[Scenario]]]:
    """Create the contents of psm files
    :param compounds: The compounds to combine with gaps to make psm files
    :param gaps: The gaps to combine with compounds to make psm files
    :param combinations: The combinations to turn into psm files
    :param crops: The crops for which psm files should be generated. If a GAP does not match the crops given, it will be
    skipped
    :param scenarios:The scenarios for which psm files should be generated. For a given GAP, the scenarios used are
    the intersection of the scenarios defined by the gap and the scenarios passed into the function
    :return: The contents of the psm files"""
    assert not (bool(compounds) ^ bool(gaps)), "Either both or neither of compound file have to be specified"
    psm_tasks: List[asyncio.Task[Tuple[str, FOCUSCrop, FrozenSet[Scenario]]]] = []
    if combinations:
        for combination in combinations:
            psm_file_scenarios = scenarios.intersection(combination.gap.defined_scenarios)
            comment = json.dumps({"combination": hash(combination)})
            psm_tasks.append(asyncio.create_task(
                _generate_psm_tuple_async(compound=combination.compound, gap=combination.gap,
                                          comment=comment, scenarios=psm_file_scenarios,
                                          pessimistic_interception=pessimistic_interception)
            ))
    if compounds and gaps:
        compounds = list(compounds)
        for gap in gaps:
            if gap.modelCrop in crops:
                for compound in compounds:
                    psm_file_scenarios = scenarios.intersection(gap.defined_scenarios)
                    comment = json.dumps({"compound": hash(compound), "gap": hash(gap)})
                    psm_tasks.append(asyncio.create_task(
                        _generate_psm_tuple_async(compound, gap, comment, psm_file_scenarios,
                                                  pessimistic_interception)
                    ))
    return [await x for x in psm_tasks]


async def _generate_psm_tuple_async(
        compound: Compound, gap: GAP, comment: str, scenarios: FrozenSet[Scenario],
        pessimistic_interception: bool = False
) -> Tuple[str, FOCUSCrop, FrozenSet[Scenario]]:
    return _generate_psm_contents(compound, gap, comment, pessimistic_interception), gap.modelCrop, scenarios


def _generate_psm_contents(compound: Compound, gap: GAP, comment: str, pessimistic_interception: bool = False) -> str:
    """For a given compound and gap file, generate the matching psm files 
    :param gap: The gap file to use when generating psm file
    :param compound: The compound file to use when generating psm file
    :param comment: The comment in the resulting psm file
    :return: The contents of the psm file"""

    psm_file = PsmFile.from_input(compound=compound, gap=gap)
    psm_file = replace(psm_file, comment=comment)
    return psm_file.render(pessimistic_interception)


def parse_args() -> Namespace:
    """Parses command line arguments"""
    parser = ArgumentParser()
    parser.add_argument('-c', '--compound-file', default=None, type=Path,
                        help='The compound to create a psm file for. If this is a directory, create psm files for '
                             'every compound file in the directory, with .json files assumed to be compound files and '
                             'no recursion')
    parser.add_argument('--combination_file', default=None, type=Path,
                        help='A file of combinations of gap and compounds. '
                             'If this is a directory, read all files in that directory')
    parser.add_argument('-g', '--gap-file', default=None, type=Path,
                        help='The gap to create a psm file for. If this is a directory, create psm files for every '
                             'gap file in the directory, with .json files assumed to be compound files and no '
                             'recursion')
    parser.add_argument('-o', '--output-dir', default=Path('output'), type=Path,
                        help='The directory for output files. '
                             'The files will be named {COMPOUND_FILE}-{GAP_FILE}-{MATURATION}-{DAY}.psm. '
                             'Defaults to a folder named output')
    parser.add_argument('--pessimistic-interception', action='store_true',
                        help='Use only the interception value of the first application')
    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger()
    jsonLogger.configure_logger_from_argparse(logger, args)
    return args


if __name__ == '__main__':
    main()
