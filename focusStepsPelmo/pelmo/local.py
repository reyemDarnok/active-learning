#!/usr/bin/env python3
"""A script for running Pelmo locally"""
import asyncio
import logging
from argparse import ArgumentParser, Namespace
from contextlib import suppress
from multiprocessing import cpu_count
from pathlib import Path
from shutil import rmtree
from typing import Awaitable, Coroutine, FrozenSet, Optional, Type, Union

from sys import path
path.append(str(Path(__file__).parent.parent.parent))
from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.ioTypes.compound import Compound
from focusStepsPelmo.ioTypes.gap import GAP, FOCUSCrop, Scenario
from focusStepsPelmo.pelmo.creator import generate_psm_files
from focusStepsPelmo.pelmo.runner import run_psms
from focusStepsPelmo.pelmo.summarize import rebuild_output_to_file
from focusStepsPelmo.util import jsonLogger
from focusStepsPelmo.util.datastructures import correct_type


def main():
    """The entry point for running this script from the command line"""
    args = parse_args()
    logger = logging.getLogger()

    logger.debug(args)
    run_local(compound_files=args.compound_file, gap_files=args.gap_file,
                          output_file=args.output_file, combination_dir=args.combined,
                          crops=frozenset(args.crop), scenarios=frozenset(args.scenario), threads=args.threads,
                          pessimistic_interception=args.pessimistic_interception)


def run_local(output_file: Path, compound_files: Optional[Path] = None, gap_files: Optional[Path] = None,
              combination_dir: Optional[Path] = None,
              crops: FrozenSet[FOCUSCrop] = frozenset(FOCUSCrop), scenarios: FrozenSet[Scenario] = frozenset(Scenario),
              threads: int = cpu_count() - 1, pessimistic_interception: bool = False):
    """Run Pelmo locally
    :param output_file: The file for the summary of results
    :param compound_files: The compounds to combine with gaps to form runs
    :param combination_dir: The combinations to form runs from
    :param gap_files: The gaps to combine with compounds to form runs
    :param crops: The crops to start runs for. Defaults to all crops
    :param scenarios: The scenarios to start runs for. Defaults to all scenarios
    :param threads: The number of threads to use when running. Defaults to using all but one CPU core of the system"""
    logger = logging.getLogger()
    logger.info(scenarios)

    logger.info('Starting to generate psm files')
    compounds = Compound.from_path(compound_files) if compound_files else None
    gaps = GAP.from_path(gap_files) if gap_files else None
    combinations = Combination.from_path(combination_dir) if combination_dir else None
    psm_files = generate_psm_files(compounds=compounds, gaps=gaps, combinations=combinations, crops=crops,
                                   scenarios=scenarios, pessimistic_interception=pessimistic_interception)
    logger.info('Starting to run Pelmo')
    results = run_psms(run_data=psm_files,
                       max_workers=threads)

    logger.info('Dumping results of Pelmo runs to %s', output_file)
    rebuild_output_to_file(file=output_file, results=results,
                           input_directories=tuple(x for x in (compound_files, gap_files, combination_dir) if x), pessimistic_interception=pessimistic_interception)

def parse_args() -> Namespace:
    """Parse command line arguments"""
    parser = ArgumentParser()
    parser.add_argument('-c', '--compound-file', default=None, type=Path,
                        help='The compound to create a psm file for. If this is a directory, create psm files for '
                             'every compound file in the directory, with .json files assumed to be compound files and '
                             'no recursion')
    parser.add_argument('-g', '--gap-file', default=None, type=Path,
                        help='The gap to create a psm file for. If this is a directory, create psm files for every '
                             'gap file in the directory, with .json files assumed to be compound files and no '
                             'recursion')
    parser.add_argument('--combined', default=None, type=Path,
                        help="Combinations of gaps and compounds. If it is a directory, parse every .json file in"
                             "that directory")
    parser.add_argument('-o', '--output-file', default=Path('output.json'), type=Path,
                        help='The name of the output file, the extension will be replaced based on the output format. '
                             'Defaults to "output.json"')
    parser.add_argument('--crop', nargs='*', default=FOCUSCrop, type=FOCUSCrop.parse,
                        help="Which crops to run. Defaults to all crops")
    parser.add_argument('-t', '--threads', type=int, default=cpu_count() - 1,
                        help="The maximum number of threads for Pelmo. Defaults to cpu_count - 1")
    parser.add_argument('-s', '--scenario', nargs='*', type=lambda x: correct_type(x, Scenario),
                        default=frozenset(Scenario),
                        help="The scenarios to simulate. Can be specified multiple times. Defaults to all scenarios. "
                             "A scenario will be calculated if it is defined both here and for the crop")
    parser.add_argument('--pessimistic-interception', action='store_true',
                        help='Use only the interception value of the first application')
    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger()

    jsonLogger.configure_logger_from_argparse(logger, args)
    return args


if __name__ == '__main__':
    main()
