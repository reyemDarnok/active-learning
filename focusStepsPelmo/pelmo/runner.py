#!/usr/bin/env python3
import csv
import json
import logging
import subprocess
from argparse import Namespace, ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from multiprocessing import cpu_count
from pathlib import Path
from shutil import copytree, rmtree
from threading import current_thread
from typing import Generator, Iterable, List, Optional, Tuple, TypeVar, Union, Sequence
from zipfile import ZipFile

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from focusStepsPelmo.ioTypes import gap
from focusStepsPelmo.ioTypes.gap import FOCUSCrop, Scenario
from focusStepsPelmo.ioTypes.pelmo import ChemPLM, PelmoResult, WaterPLM
from focusStepsPelmo.pelmo.summarize import rebuild_output_to_file
from focusStepsPelmo.util import conversions
from focusStepsPelmo.util import jsonLogger

jinja_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=select_autoescape(),
                        undefined=StrictUndefined)


def _init_thread(working_dir: Path):
    """Initialise a working directory for the current thread in the overarching working directory.
    This mostly consists of copying reference files
    :param working_dir: Where to create the subdirectory for the thread"""
    # PELMO can't run multiple times in the same directory at the same time
    logger = logging.getLogger()

    logger.debug('Starting initialisation of %s', current_thread().name)
    runner_dir = working_dir / current_thread().name
    sample_dir = working_dir / 'sample'
    copytree(sample_dir, runner_dir)
    logger.info('%s is initialised', current_thread().name)


def main():
    args = parse_args()
    logger = logging.getLogger()
    logger.debug(args)
    files = list(args.psm_files.glob('*.psm') if args.psm_files.is_dir() else [args.psm_files])
    logging.info('Running for the following psm files: %s', files)
    write_psm_results(args.output, files, None, working_dir=args.working_dir, crops=args.crop, scenarios=args.scenario,
                      max_workers=args.threads)


def write_psm_results(output_file: Path, psm_files: Iterable[Union[Path, str]],
                      input_directories: Optional[Tuple[Path]] = None, working_dir: Path = Path.cwd() / 'pelmo',
                      crops: Iterable[FOCUSCrop] = FOCUSCrop, scenarios: Iterable[Scenario] = Scenario,
                      max_workers: int = cpu_count() - 1):
    results = run_psms(psm_files=psm_files, working_dir=working_dir,
                       crops=crops, scenarios=scenarios, max_workers=max_workers)
    if input_directories:
        rebuild_output_to_file(file=output_file, results=results, input_directories=input_directories)
    else:
        with output_file.open('w') as output:
            if output_file.suffix == '.json':
                json.dump(list(results), output, cls=conversions.EnhancedJSONEncoder)
            else:
                writer = csv.writer(output)
                writer.writerows((result.psm_comment, result.crop, result.scenario, result.pec) for result in results)


def _make_runs(psm_files: Iterable[Union[Path, str]], crops: Iterable[FOCUSCrop], scenarios: Iterable[Scenario]) -> \
        Generator[Tuple[Union[Union[Path, str], FOCUSCrop, Scenario]], None, None]:
    crops = list(crops)
    scenarios = list(scenarios)
    for psm_file in psm_files:
        for crop in crops:
            for scenario in scenarios:
                yield psm_file, crop, scenario


T = TypeVar('T')


def repeat_infinite(value: T) -> Generator[T, None, None]:
    while True:
        yield value


def run_psms(psm_files: Iterable[Union[Path, str]], working_dir: Path,
             crops: Iterable[FOCUSCrop] = FOCUSCrop, scenarios: Iterable[Scenario] = Scenario,
             max_workers: int = cpu_count() - 1) -> Generator[PelmoResult, None, None]:
    """Run all given psm_files using working_dir as scratch space. 
    When given scenarios that are not defined for some given crops, they are silently ignored for those crops only
    :param psm_files: The files to run
    :param working_dir: Where to run them
    :param crops: The crops to run. Crop / scenario combinations that are not defined are silently skipped
    :param scenarios: The scenarios to run. Scenario / crop combinations that are not defined are silently skipped
    :param max_workers: How many worker threads to use at most
    :return: A Generator of the results of the calculations. Makes new results available as their calculations finish.
                No particular ordering is guaranteed but the calculations are started in order of
                psm_file, then crop, then scenario"""
    with suppress(FileNotFoundError):
        rmtree(working_dir)
    extract_zip(working_dir / 'sample', Path(__file__).parent / 'data' / 'FOCUS.zip')
    pool = ThreadPoolExecutor(max_workers=max_workers, initializer=_init_thread, initargs=(working_dir,))
    yield from pool.map(single_pelmo_run, _make_runs(psm_files=psm_files, crops=crops, scenarios=scenarios),
                        repeat_infinite(working_dir))
    pool.shutdown()


def single_pelmo_run(run_data: Tuple[Union[Path, str], FOCUSCrop, Scenario], working_dir: Path) -> PelmoResult:
    """Runs a single psm/crop/scenario combination.
    Assumes that it is in a multithreading context after _init_thread as run
    :param working_dir: Where to find the scenario data
    :param run_data: The input to Pelmo
    :return: The result of the Pelmo run"""
    logger = logging.getLogger()
    psm_file, crop, scenario = run_data
    inp_file_template = jinja_env.get_template('pelmo.inp.j2')
    dat_file_template = jinja_env.get_template('input.dat')
    scenario_dirs = working_dir / current_thread().name
    scenario_dir = scenario_dirs / scenario.value
    if isinstance(psm_file, Path):
        run_dir = scenario_dir / f'{psm_file.stem}.run'
    else:
        run_dir = scenario_dir / f'{hash(psm_file)}.run'
    crop_dir = run_dir / crop.display_name

    logger.debug('Creating run directory %s', crop_dir)
    crop_dir.mkdir(exist_ok=True, parents=True)
    if isinstance(psm_file, Path):
        target_psm_file = crop_dir / psm_file.name
    else:
        target_psm_file = crop_dir / f"{hash(psm_file)}.psm"
    target_inp_file = crop_dir / 'pelmo.inp'
    target_dat_file = crop_dir / 'input.dat'

    logger.debug('Creating pelmo input files')
    if isinstance(psm_file, Path):
        target_psm_file.write_text(psm_file.read_text())
    else:
        target_psm_file.write_text(psm_file)
        psm_file = target_psm_file
    target_inp_file.write_text(inp_file_template.render(psm_file=psm_file, crop=crop, scenario=scenario))
    target_dat_file.write_text(dat_file_template.render())

    logger.info('Starting PELMO run for compound: %s :: crop: %s :: scenario: %s', target_psm_file.stem,
                crop.display_name, scenario.value)
    try:
        process = subprocess.run([str((Path(__file__).parent / 'data' / 'PELMO500.EXE').absolute())], cwd=crop_dir,
                                 check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise ValueError(
            f"Pelmo exited prematurely with an error while calculating {psm_file.name} {crop.display_name} "
            f"{scenario.value}. Pelmo returned exitcode {e.returncode}. The Pelmo error was {e.output}")
    logger.info('Finished PELMO run for %s, %s and %s', target_psm_file.stem, crop.display_name, scenario.value)
    if b"F A T A L   E R R O R" in process.stdout:
        raise ValueError(
            f"Pelmo completed with error while calculating {psm_file.name} {crop.display_name} {scenario.value}. "
            f"The Pelmo output was {process.stdout.decode(errors='backslashreplace')}")

    with psm_file.open() as psm:
        psm.readline()
        psm_comment = psm.readline()

    result = PelmoResult(psm_comment=psm_comment, scenario=scenario, crop=crop, pec=parse_pelmo_result(crop_dir))
    if logger.level > logging.DEBUG:
        rmtree(crop_dir)
    return result


def parse_pelmo_result(run_dir: Path, target_compartment=21) -> Tuple[float, ...]:
    """Parses the Pelmo output files to determine the PEC that pelmo calculated
    :param run_dir: Where Pelmo was executed and placed its result files
    :param target_compartment: Which compartment to take as result
    :return: A list of concentrations, element 0 is the parent
    and the others are the Metabolites in the order defined by Pelmo"""
    water_file = run_dir / "WASSER.PLM"
    chem_files = run_dir.glob("CHEM*.PLM")

    water_plm = WaterPLM(water_file)
    results = []
    for chem_file in chem_files:
        chem_plm = ChemPLM(chem_file)
        chem_horizons = [horizon for year in chem_plm.horizons for horizon in year if
                         horizon.compartment == target_compartment]
        water_horizons = [horizon for year in water_plm.horizons for horizon in year if
                          horizon.compartment == target_compartment]
        # mass in g/ha / water in mm
        # input is in kg/ha and cm
        pecs = [(chem_horizons[i].leaching_output * 1000 / (water_horizons[i].leaching_output * 10) * 100) if
                water_horizons[i].leaching_output > 0 else 0 for i in range(len(chem_horizons))]
        pecs = pecs[6:]
        pecs.sort()
        percentile = 0.8
        lower = int((len(pecs) - 1) * percentile)
        pec_groundwater = (pecs[lower] + pecs[lower + 1]) / 2
        results.append(pec_groundwater)

    return tuple(results)


def extract_zip(working_dir: Path, focus_zip: Path):
    logger = logging.getLogger()

    logger.info(f'Extracting {focus_zip.name} to {working_dir.name}')
    with ZipFile(focus_zip) as zip_file:
        zip_file.extractall(path=working_dir)
    logger.info('Finished extracting zip')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-p', '--psm-files', type=Path, required=True,
                        help="The psm file to run. If this is a directory, run all .psm files in this directory")
    parser.add_argument('-w', '--working-dir', type=Path, default=Path.cwd() / 'pelmofiles',
                        help="The directory to use as a root working directory. "
                             "Will be filled with expanded zips and defaults to the current working directory")
    parser.add_argument('-f', '--focus-dir', type=Path, default=Path(__file__).parent / 'data' / 'Focus.zip',
                        help="The PELMO FOCUS directory to use. "
                             "If a zip, will be unpacked first. Defaults to a bundled zip.")
    parser.add_argument('-e', '--pelmo-exe', type=Path,
                        default=Path(__file__).parent / 'data' / 'PELMO500.exe',
                        help="The PELMO executable to use for running. "
                             "Defaults to a bundled PELMO installation. "
                             "This should point to the CLI EXE, usually named PELMO500.EXE "
                             "NOT to the GUI EXE usually named wpelmo.exe.")
    parser.add_argument('-c', '--crop', nargs='*',
                        type=gap.FOCUSCrop.from_acronym, default=list(gap.FOCUSCrop),
                        help="The crops to simulate. Can be specified multiple times. "
                             "Should be listed as a two letter acronym. "
                             "The selected crops have to be present in the FOCUS zip, "
                             "the bundled zip includes all crops. Defaults to all crops.")
    parser.add_argument('-s', '--scenario', nargs='*', type=lambda x: conversions.str_to_enum(x, Scenario),
                        default=list(gap.Scenario),
                        help="The scenarios to simulate. Can be specified multiple times. Defaults to all scenarios. "
                             "A scenario will be calculated if it is defined both here and for the crop")
    parser.add_argument('-t', '--threads', type=int, default=cpu_count() - 1,
                        help="The maximum number of threads for Pelmo. Defaults to cpu_count - 1")
    parser.add_argument('-o', '--output', type=Path, default=Path('output.json'),
                        help="Where to write the results to. Defaults to output.json")
    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger()

    jsonLogger.configure_logger_from_argparse(logger, args)
    return args


if __name__ == '__main__':
    main()
