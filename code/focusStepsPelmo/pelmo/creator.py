#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
import logging
from pathlib import Path
from typing import Generator, Iterable, List, Type, TypeVar, Union
from jinja2 import Environment, FileSystemLoader, select_autoescape, StrictUndefined
import json
import sys 
sys.path += [str(Path(__file__).parent.parent)]
from ioTypes.combination import Combination
from util.conversions import EnhancedJSONEncoder
from psm_file import PsmFile
import util.jsonLogger as jsonLogger
from ioTypes.compound import Compound
from ioTypes.gap import GAP



jinja_env = Environment(loader=FileSystemLoader([Path(__file__).parent / "templates", Path(__file__).parent / "templates" / "psm-fragments"]), 
                        autoescape=select_autoescape(), undefined=StrictUndefined)

def main():
    args = parse_args()
    generate_psm_files(output_dir=args.output_dir, compound_file=args.compound_file, gap_file=args.gap_file)

def write_psm_files(output_dir: Path, 
                    compounds: Iterable[Union[Path, Compound]] = None,
                    gaps: Iterable[Union[Path, Compound]] = None,
                    combinations: Iterable[Union[Path, Compound]] = None) -> int:
    if compounds:
        compounds = load_or_use(compounds, Compound)
    if gaps:
        gaps = load_or_use(gaps, GAP)
    if combinations:
        combinations = load_or_use(combinations, Combination)

T = TypeVar('T')
def load_or_use(it: Iterable[Union[Path, T]], t: Type[T]) -> Generator[T, None, None]:
    for element in it:
        if isinstance(element, t):
            yield element
        else:
            yield from load_class(element, t)


def load_class(source: Path, t: Type[T]) -> List[T]:
    if source.suffix == '.json':
        with source.open() as file:
            json_content = json.load(file)
        if isinstance(json_content, list):
            return [t(**element) for element in json_content]
        else:
            return [t(**json_content)]
    elif source.suffix == 'xlsx':
        return t.from_excel(source)

def generate_psm_files(output_dir: Path, compound_file: Path = None, gap_file: Path = None, combination_dir: Path = None) -> int:
    '''Creates the crossproduct of compound and gap files and saves the resulting psm files in output_dir
    :param output_dir: Where to save the .psm files. The files will be named {COMPOUND_FILE}-{GAP_FILE}-{MATURATION}-{DAY}.psm
    :param compound_file: Either a compound file or a directory filled with compound.json files. If a directory all *.json files are assumed to be compound files
    :param gap_file: Either a gap file or a directory filled with gap.json files. If a directory all *.json files are assumed to be gap files
    :return: How many files were generated'''
    assert not (bool(compound_file) ^ bool(gap_file)), "Either both or neither of compound file have to be specified"
    output_dir.mkdir(exist_ok=True, parents=True)
    total = 0
    if combination_dir:
        if combination_dir.is_dir():
            combinations = combination_dir.glob('*.json')
        elif combination_dir.exists():
            combinations = [combination_dir]
        else:
            combinations = []
        for combination_file in combinations:
            with combination_file.open() as fp:
                combination = Combination(**json.load(fp))
            output_file = output_dir / f"{hash(combination_file)}.psm"
            comment = json.dumps({"combination": str(combination_file)})
            total += 1
            output_file.write_text(_generate_psm_contents(combination.compound, combination.gap, comment))
    if compound_file and gap_file:
        if compound_file.is_dir():
            compounds = compound_file.glob('*.{json,xslx}')
        elif compound_file.exists():
            compounds = [compound_file]
        else:
            compounds = []
        if gap_file.is_dir():
            gaps = list(gap_file.glob('*.{json,xslx}'))
        elif gap_file.exists():
            gaps = [gap_file]
        else:
            gaps = []

        compounds = [compound for compound_group in compounds for compound in load_compound(compound_group)]
        gaps = [gap for gap_group in gaps for gap in load_gap(gap_group)]
        for compound in compounds:
            for gap in gaps:
                output_file = output_dir / f"{hash(compound)}-{hash(gap)}.psm"
                comment = json.dumps({"compound": str(compound_file), "gap": str(gap_file)})
                total += 1
                output_file.write_text(_generate_psm_contents(compound, gap, comment))
    return total

def load_gap(gap: Path) -> List[GAP]:
    if gap.suffix == '.json':
        with gap.open() as file:
            json_content = json.load(file)
        if isinstance(json_content, list):
            return [GAP(**element) for element in json_content]
        else:
            return [GAP(**json_content)]
    elif gap.suffix == '.xlsx':
        return GAP.from_excel(gap)

def load_compound(compound: Path) -> List[Compound]:
    if compound.suffix == '.json':
        with compound.open() as file:
            json_content = json.load(file)
        if isinstance(json_content, list):
            return [Compound(**element) for element in json_content]
        else:
            return [Compound(**json_content)]
    elif compound.suffix == '.xlsx':
        return Compound.from_excel(compound) 

def _generate_psm_contents(compound: Compound, gap: GAP, comment: str) -> str:
    '''For a given compound and gap file, generate the matching psm files 
    :param gap_file: The gap file to use when generating psm file
    :param compound_file: The compound file to use when generating psm file
    :return: The contents of the psm file'''
    
    psm_file = PsmFile.fromInput(compound=compound, gap=gap)
    psm_template = jinja_env.get_template('general.psm.j2')
    psm_file.comment = comment
    return psm_template.render(**psm_file._asdict())

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-c', '--compound-file', required=True, type=Path, help='The compound to create a psm file for. If this is a directory, create psm files for every compound file in the directory, with .json files assumed to be compound files and no recursion')
    parser.add_argument('-g', '--gap-file', required=True, type=Path, help='The gap to create a psm file for. If this is a directory, create psm files for every gap file in the directory, with .json files assumed to be compound files and no recursion')
    parser.add_argument('-o', '--output-dir', default=Path('output'), type=Path, help='The directory for output files. The files will be named {COMPOUND_FILE}-{GAP_FILE}-{MATURATION}-{DAY}.psm. Defaults to a folder named output')
    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger()
    jsonLogger.configure_logger_from_argparse(logger, args)
    return args

if __name__ == '__main__':
    main()