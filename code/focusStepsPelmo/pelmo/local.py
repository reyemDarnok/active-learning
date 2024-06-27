#!/usr/bin/env python3
import logging
from argparse import ArgumentParser, Namespace
from contextlib import suppress
from pathlib import Path
import sys
from typing import Sequence
sys.path += [str(Path(__file__).parent.parent)]
from pelmo.summarize import rebuild_output_to_file
from util import conversions
from ioTypes.gap import FOCUSCrop, Scenario
from pelmo.creator import generate_psm_files
from pelmo.runner import run_psms
from shutil import rmtree
from util import jsonLogger

from multiprocessing import cpu_count
import json


def main():
    args = parse_args()
    logger = logging.getLogger()

    logger.debug(args)
    run_local(work_dir=args.work_dir, compound_files=args.compound_file, gap_files=args.gap_file, output_file=args.output_file,
              crops=args.crop, scenarios=args.scenario, threads=args.threads)

def run_local(work_dir: Path, output_file: Path, compound_files: Path = None, gap_files: Path = None, combination_dir: Path = None, 
              crops: Sequence[FOCUSCrop]=FOCUSCrop, scenarios: Sequence[Scenario]=Scenario, threads: int = cpu_count() - 1):
    logger = logging.getLogger()
    with suppress(FileNotFoundError): rmtree(work_dir)
    psm_dir: Path = work_dir / 'psm'
    psm_dir.mkdir(exist_ok=True, parents=True)
    focus_dir: Path = work_dir / 'FOCUS'
    focus_dir.mkdir(exist_ok=True, parents=True)

    logger.info('Starting to generate psm files')
    psm_files = generate_psm_files(compound_file=compound_files, gap_file=gap_files, combination_dir=combination_dir)

    logger.info('Starting to run Pelmo')
    results = run_psms(psm_files=psm_files, working_dir=focus_dir,crops=crops, scenarios=scenarios, max_workers=threads)
    
    logger.info('Dumping results of Pelmo runs to %s', output_file)
    rebuild_output_to_file(output_file, [x for x in (compound_files, gap_files, combination_dir) if x], results)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-c', '--compound-file', required=True, type=Path, help='The compound to create a psm file for. If this is a directory, create psm files for every compound file in the directory, with .json files assumed to be compound files and no recursion')
    parser.add_argument('-g', '--gap-file', required=True, type=Path, help='The gap to create a psm file for. If this is a directory, create psm files for every gap file in the directory, with .json files assumed to be compound files and no recursion')
    parser.add_argument('-w', '--work-dir', default=Path.cwd() / 'pelmofiles', type=Path, help='The directory in which files for Pelmo will be created. Defaults to the current directory')
    parser.add_argument('-o', '--output-file', default=Path('output.ext'), type=Path, help='The name of the output file, the extension will be replaced based on the output format. Defaults to "output.ext"')
    parser.add_argument(      '--crop', nargs='*', default=FOCUSCrop, type=FOCUSCrop.from_acronym, help="Which crops to run. Defaults to all crops")
    parser.add_argument('-t', '--threads', type=int, default=cpu_count() - 1, help="The maximum number of threads for Pelmo. Defaults to cpu_count - 1")
    parser.add_argument('-s', '--scenario', nargs='*', type=lambda x: conversions.str_to_enum(x, Scenario), default=list(Scenario), help="The scenarios to simulate. Can be specified multiple times. Defaults to all scenarios. A scenario will be calculated if it is defined both here and for the crop")
    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger()

    jsonLogger.configure_logger_from_argparse(logger, args)
    return args

if __name__ == '__main__':
    main()