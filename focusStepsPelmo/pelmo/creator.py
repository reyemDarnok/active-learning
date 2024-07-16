#!/usr/bin/env python3
import json
import logging
from argparse import ArgumentParser, Namespace
from dataclasses import replace
from pathlib import Path
from typing import Generator, Iterable, Type, TypeVar, Union, Tuple, Set

from jinja2 import Environment, select_autoescape, StrictUndefined, PackageLoader

from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.ioTypes.compound import Compound
from focusStepsPelmo.ioTypes.gap import GAP, FOCUSCrop, Scenario
from focusStepsPelmo.pelmo.psm_file import PsmFile
from focusStepsPelmo.util import jsonLogger as jsonLogger

jinja_env = Environment(loader=PackageLoader('focusStepsPelmo.pelmo'),
                        autoescape=select_autoescape(), undefined=StrictUndefined)


def main():
    args = parse_args()
    write_psm_files(output_dir=args.output_dir, compounds=args.compound_file, gaps=args.gap_file,
                    combinations=args.combination_file)


def write_psm_files(output_dir: Path,
                    compounds: Iterable[Union[Path, Compound]] = None,
                    gaps: Iterable[Union[Path, Compound]] = None,
                    combinations: Iterable[Union[Path, Compound]] = None) -> int:
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
    for psm_file in generate_psm_files(compounds=compounds, gaps=gaps, combinations=combinations):
        total += 1
        (output_dir / f"{hash(psm_file)}.psm").write_text(psm_file[0])
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
            yield from load_class(element, t)


def load_class(source: Path, t: Type[T]) -> Generator[T, None, None]:
    """Given a path, load the object in the class
    :param source: The source path for the object
    :param t: The type of the object in source
    :return: All objects of type t that could be parsed from source"""
    if source.suffix == '.json':
        with source.open() as file:
            json_content = json.load(file)
        if isinstance(json_content, list):
            for element in json_content:
                yield t(**element)
        else:
            yield t(**json_content)
    elif source.suffix == 'xlsx':
        yield from t.from_excel(source)


def generate_psm_files(compounds: Iterable[Compound] = None, gaps: Iterable[GAP] = None,
                       crops: Set[FOCUSCrop] = None, scenarios: Set[Scenario] = None,
                       combinations: Iterable[Combination] = None) -> Generator[
    Tuple[str, Set[FOCUSCrop], Set[Scenario]], None, None]:
    """Create the contents of psm files
    :param compounds: The compounds to combine with gaps to make psm files
    :param gaps: The gaps to combine with compounds to make psm files
    :param combinations: The combinations to turn into psm files
    :return: The contents of the psm files"""
    if scenarios is None:
        scenarios = set(Scenario)
    if crops is None:
        crops = set(FOCUSCrop)
    assert not (bool(compounds) ^ bool(gaps)), "Either both or neither of compound file have to be specified"
    if combinations:
        for combination in combinations:
            comment = json.dumps({"combination": hash(combination)})
            yield _generate_psm_contents(compound=combination.compound, gap=combination.gap, comment=comment)
    if compounds and gaps:
        gaps = list(gaps)
        for compound in compounds:
            for gap in gaps:
                psm_file_crops = crops.intersection(gap.modelCrops)
                psm_file_scenarios = scenarios.intersection(gap.defined_scenarios)
                comment = json.dumps({"compound": hash(compound), "gap": hash(gap)})
                yield _generate_psm_contents(compound, gap, comment), psm_file_crops, psm_file_scenarios


def _generate_psm_contents(compound: Compound, gap: GAP, comment: str) -> str:
    """For a given compound and gap file, generate the matching psm files 
    :param gap: The gap file to use when generating psm file
    :param compound: The compound file to use when generating psm file
    :param comment: The comment in the resulting psm file
    :return: The contents of the psm file"""

    psm_file = PsmFile.from_input(compound=compound, gap=gap)
    psm_file = replace(psm_file, comment=comment)
    return psm_file.render()


def parse_args() -> Namespace:
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
    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger()
    jsonLogger.configure_logger_from_argparse(logger, args)
    return args


if __name__ == '__main__':
    main()
