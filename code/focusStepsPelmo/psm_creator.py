#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
import logging
import jsonLogger
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape, StrictUndefined
from enum import Enum
import json
from focusStepsDatatypes import compound, gap
jinja_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=select_autoescape(), undefined=StrictUndefined)

class ApplicationType(Enum):
    soil = 1
    linear = 2
    exp_foliar = 3
    manual = 4

def main():
    args = parse_args()
    generate_psm_files(output_dir=args.output_dir, compound_file=args.compound_file, gap_file=args.gap_file)

def generate_psm_files(output_dir: Path, compound_file: Path, gap_file: Path):
    '''Creates the crossproduct of compound and gap files and saves the resulting psm files in output_dir'''
    output_dir.mkdir(exist_ok=True, parents=True)
    if compound_file.is_dir():
        if gap_file.is_dir():
            for actual_compound_file in compound_file.glob('*.json'):
                for actual_gap_file in gap_file.glob('*.json'):
                    _generate_psms(actual_compound_file, actual_gap_file, output_dir)
        else:
            for actual_compound_file in compound_file.glob('*.json'):
                _generate_psms(actual_compound_file, gap_file, output_dir)
    else:
        if gap_file.is_dir():
            for actual_gap_file in gap_file.glob('*.json'):
                _generate_psms(compound_file, actual_gap_file, output_dir)
        else:
            _generate_psms(compound_file, gap_file, output_dir)

def _generate_psms(compound_file: Path, gap_file: Path, output_dir: Path):
    '''For a given compound and gap file, generate the matching psm files and write them to output_dir'''
    psm_compound = compound.Substance(**json.loads(compound_file.read_text()))
    psm_gap = gap.GAP(**json.loads(gap_file.read_text()))
    psm_template = jinja_env.get_template('general.psm.j2')
    comment = f"Generated from {compound_file.stem} and {gap_file.stem}"
    for timing in psm_gap.timings:
        psm_content = psm_template.render(compound=psm_compound, 
                                gap=psm_gap,
                                scenario={
                                            "plant_decay_rate": 0.0693,
                                            "washoff": 1,
                                            "penetration": 0.0693,
                                            "photodegradation": 0,
                                            "irradiance": 500,
                                            "laminar_layer": 1000},
                                comment=comment, 
                                timing = timing,
                                henry_calc=True,
                                kd_calc=True,
                                time=0,
                                Ffield=0,
                                frpex=0,
                                pesticide={'application_type': ApplicationType.manual},
                                )
        (output_dir / f'{compound_file.stem}-{gap_file.stem}-{timing.emergence.name}-{timing.offset}.psm').write_text(psm_content)

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