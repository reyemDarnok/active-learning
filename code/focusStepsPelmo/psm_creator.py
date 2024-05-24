#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
from jinja2 import Environment, PackageLoader, select_autoescape, StrictUndefined
from enum import Enum
import json
from focusStepsDatatypes import compound, gap, scenario

logging.basicConfig(level=logging.DEBUG)
jinja_env = Environment(loader=PackageLoader("main"), autoescape=select_autoescape(), undefined=StrictUndefined)
args = Namespace()

class Location(Enum):
    Châteaudun = 'C'
    Hamburg = 'H'
    Kremsmünster = 'K'
    Okehampton = 'N'
    Piacenza = 'P'
    Porto = 'O'
    Sevilla = 'S'
    Thiva = 'T'

def main():
    parse_args()
    print(args)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    if args.compound_file.is_dir():
        if args.gap_file.is_dir():
            for compound_file in args.compound_file.glob('*.json'):
                for gap_file in args.gap_file.glob('*.json'):
                    generate_psm(compound_file, gap_file)
        else:
            for compound_file in args.compound_file.glob('*.json'):
                generate_psm(compound_file, args.gap_file)
    else:
        if args.gap_file.is_dir():
            for gap_file in args.gap_file.glob('*.json'):
                generate_psm(args.compound_file, gap_file)
        else:
            generate_psm(args.compound_file, args.gap_file)

def generate_psm(compound_file: Path, gap_file: Path):
    psm_compound = compound.Substance(**json.loads(compound_file.read_text()))
    psm_gap = gap.GAP(**json.loads(gap_file.read_text()))
    psm_scenario = scenario.Scenario()
    psm_template = jinja_env.get_template('general.psm.j2')
    comment = f"Generated from {compound_file.name} and {gap_file.name}"
    psm_content = psm_template.render(compound=psm_compound, 
                              gap=psm_gap,
                              scenario={"interception": 50},
                              comment=comment, 
                              locations=args.location,
                              henry_calc=True,
                              kd_calc=True,
                              time=0,
                              Ffield=0,
                              frpex=0)
    (args.output_dir / f'{compound_file.stem}-{gap_file.stem}.psm').write_text(psm_content)

def parse_args():
    global args
    parser = ArgumentParser()
    parser.add_argument('-c', '--compound-file', required=True, type=Path, help='The compound to create a psm file for. If this is a directory, create psm files for every compound file in the directory, with .json files assumed to be compound files and no recursion')
    parser.add_argument('-l', '--location', nargs='*', type=Location, default=list(Location), help='The locations to calculate. Defaults to all locations')
    parser.add_argument('-g', '--gap-file', required=True, type=Path, help='The gap to create a psm file for. If this is a directory, create psm files for every gap file in the directory, with .json files assumed to be compound files and no recursion')
    parser.add_argument('-o', '--output-dir', required=True, type=Path, help='The directory for output files. The files will be named {COMPOUND_FILE}-{GAP_FILE}.psm')
    args = parser.parse_args()

if __name__ == '__main__':
    main()