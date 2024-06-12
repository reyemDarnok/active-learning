#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
import logging
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape, StrictUndefined
import json
import sys 
sys.path += [str(Path(__file__).parent.parent)]
from psm_file import PsmFile
from ioTypes import compound, gap
import util.jsonLogger as jsonLogger



jinja_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=select_autoescape(), undefined=StrictUndefined)

def main():
    args = parse_args()
    generate_psm_files(output_dir=args.output_dir, compound_file=args.compound_file, gap_file=args.gap_file)

def generate_psm_files(compound_file: Path, gap_file: Path, output_dir: Path):
    '''Creates the crossproduct of compound and gap files and saves the resulting psm files in output_dir
    :param output_dir: Where to save the .psm files. The files will be named {COMPOUND_FILE}-{GAP_FILE}-{MATURATION}-{DAY}.psm
    :param compound_file: Either a compound file or a directory filled with compound.json files. If a directory all *.json files are assumed to be compound files
    :param gap_file: Either a gap file or a directory filled with gap.json files. If a directory all *.json files are assumed to be gap files'''
    output_dir.mkdir(exist_ok=True, parents=True)
    if compound_file.is_dir():
        if gap_file.is_dir():
            for actual_compound_file in compound_file.glob('*.json'):
                for actual_gap_file in gap_file.glob('*.json'):
                    _generate_psm(actual_compound_file, actual_gap_file, output_dir)
        else:
            for actual_compound_file in compound_file.glob('*.json'):
                _generate_psm(actual_compound_file, gap_file, output_dir)
    else:
        if gap_file.is_dir():
            for actual_gap_file in gap_file.glob('*.json'):
                _generate_psm(compound_file, actual_gap_file, output_dir)
        else:
            _generate_psm(compound_file, gap_file, output_dir)

def _generate_psm(compound_file: Path, gap_file: Path, output_dir: Path):
    '''For a given compound and gap file, generate the matching psm files and write them to output_dir
    :param output_dir: Where to save the .psm files. The files will be named {COMPOUND_FILE}-{GAP_FILE}.psm
    :param gap_file: The gap file to use when generating psm files
    :param compound_file: The compound file to use when generating psm files'''
    with compound_file.open() as fp:
        psm_compound = compound.Compound(**json.load(fp))
    with gap_file.open() as fp:
        psm_gap = gap.GAP(**json.load(fp))
    psm_file = PsmFile.fromInput(compound=psm_compound, gap=psm_gap)
    psm_template = jinja_env.get_template('general.psm.j2')
    psm_file.comment = json.dumps({"compound": str(compound_file), "gap": str(gap_file)})
    psm_content = psm_template.render(**asdict(psm_file))
    (output_dir / f'{compound_file.stem}-{gap_file.stem}.psm').write_text(psm_content)

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