#!/usr/bin/env python3
import jsonLogger
import logging
from argparse import ArgumentParser, Namespace
from contextlib import suppress
from pathlib import Path
from focusStepsDatatypes import helperfunctions
from focusStepsDatatypes.gap import PelmoCrop, Scenario
from psm_creator import generate_psm_files
from psm_runner import extract_zip, run_psms
from shutil import copytree, rmtree
from multiprocessing import cpu_count
from dataclasses import asdict
import json

logger = logging.getLogger()

def main():
    args = parse_args()
    logger.debug(args)
    logger.info('Deleting old artefacts')
    with suppress(FileNotFoundError): rmtree(args.work_dir)
    psm_dir: Path = args.work_dir / 'psm'
    psm_dir.mkdir(exist_ok=True, parents=True)
    focus_dir: Path = args.work_dir / 'FOCUS'
    focus_dir.mkdir(exist_ok=True, parents=True)
    logger.info('Starting to generate psm files')
    generate_psm_files(output_dir=psm_dir, compound_file=args.compound_file, gap_file=args.gap_file)
    logger.info('Starting to setup for running Pelmo')
    if args.focus.is_dir():
        copytree(args.focus, focus_dir / 'sample' )
    else:
        extract_zip(focus_dir / 'sample', args.focus)
    logger.info('Starting to run Pelmo')
    results = run_psms(psm_files=psm_dir.glob('*.psm'), working_dir=focus_dir, crops=args.crop, scenarios=args.scenario, pelmo_exe=args.pelmo_exe, max_workers=args.threads)
    logger.info('Dumping results of Pelmo runs to %s', args.output_file)
    args.output_file.write_text(json.dumps(results))

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-c', '--compound-file', required=True, type=Path, help='The compound to create a psm file for. If this is a directory, create psm files for every compound file in the directory, with .json files assumed to be compound files and no recursion')
    parser.add_argument('-g', '--gap-file', required=True, type=Path, help='The gap to create a psm file for. If this is a directory, create psm files for every gap file in the directory, with .json files assumed to be compound files and no recursion')
    parser.add_argument('-w', '--work-dir', default=Path().cwd(), type=Path, help='The directory in which files for Pelmo will be created. Defaults to the current directory')
    parser.add_argument('-o', '--output-file', default=Path('output.json'), type=Path, help='Where to output the collected results of the Pelmo runs. Defaults to "output.json"')
    parser.add_argument(      '--crop', nargs='*', default=list(PelmoCrop), type=PelmoCrop.from_acronym, help="Which crops to run. Defaults to all crops")
    parser.add_argument('-e', '--pelmo-exe', type=Path, default=Path('C:/FOCUS_PELMO.664') / 'PELMO500.exe', help="The PELMO executable to use for running. Defaults to the default PELMO installation. This should point to the CLI EXE, usually named PELMO500.EXE NOT to the GUI EXE usually named wpelmo.exe.")
    parser.add_argument('-t', '--threads', type=int, default=cpu_count() - 1, help="The maximum number of threads for Pelmo. Defaults to cpu_count - 1")
    parser.add_argument('-s', '--scenario', nargs='*', type=lambda x: helperfunctions.str_to_enum(x, Scenario), default=list(Scenario), help="The scenarios to simulate. Can be specified multiple times. Defaults to all scenarios. A scenario will be calculated if it is defined both here and for the crop")
    parser.add_argument('-f', '--focus', type=Path, default=Path('FOCUS.zip', help="The Focus data to use. Unzips it if it is a zip. Defaults to a bundled zip"))
    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    jsonLogger.configure_logger_from_argparse(logger, args)

if __name__ == '__main__':
    main()