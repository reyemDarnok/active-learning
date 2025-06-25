#!/usr/bin/env python3
"""A file for methods related to directly running pelmo"""
import csv
import json
import logging
import shutil
import subprocess
from argparse import Namespace, ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from multiprocessing import cpu_count
from pathlib import Path
from shutil import copytree, rmtree
import tempfile
from threading import current_thread
from typing import Generator, Iterable, Optional, Tuple, Union, Dict, FrozenSet
from zipfile import ZipFile

from jinja2 import Environment, StrictUndefined, select_autoescape, PackageLoader

from focusStepsPelmo.ioTypes import gap
from focusStepsPelmo.ioTypes.gap import FOCUSCrop, Scenario
from focusStepsPelmo.ioTypes.pelmo import ChemPLM, PelmoResult, WaterPLM, lookup_crop_file_name
from focusStepsPelmo.pelmo.summarize import rebuild_output_to_file
from focusStepsPelmo.util import conversions
from focusStepsPelmo.util import jsonLogger
from focusStepsPelmo.util.datastructures import correct_type
from focusStepsPelmo.util.iterable_helper import repeat_infinite

jinja_env = Environment(loader=PackageLoader('focusStepsPelmo.pelmo'),
                        autoescape=select_autoescape(), undefined=StrictUndefined)


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
    """The entry point for running this script from the command line"""
    args = parse_args()
    logger = logging.getLogger()
    logger.debug(args)
    files: Iterable[Path] = args.psm_files.glob('*.psm') if args.psm_files.is_dir() else [args.psm_files]
    logging.info('Running for the following psm files: %s', files)
    crops: FrozenSet[FOCUSCrop] = frozenset(args.crop)
    scenarios: FrozenSet[Scenario] = frozenset(args.scenario)
    run_data = [(file, crop, scenarios) for file in files for crop in crops]
    write_psm_results(output_file=args.output, run_data=run_data,
                      max_workers=args.threads, pessimistic_interception=args.pessimistic_interception)


def write_psm_results(output_file: Path,
                      run_data: Iterable[Tuple[Union[Path, str], FOCUSCrop, FrozenSet[Scenario]]], pessimistic_interception: bool,
                      input_directories: Optional[Tuple[Path]] = None,
                      max_workers: int = cpu_count() - 1, ):
    """Run Pelmo and write the results to output_file
    :param output_file: Where to write the result to
    :param run_data: What to run. A List of (psm file, crop, scenarios) tuples
    :param input_directories: In which directories to find the input files for correctly regenerating the input
    information
    :param max_workers: The maximum of threads to use for running Pelmo"""
    results = run_psms(run_data=run_data,
                       max_workers=max_workers)
    if input_directories:
        rebuild_output_to_file(file=output_file, results=results, input_directories=input_directories, pessimistic_interception=pessimistic_interception)
    else:
        with output_file.open('w') as output:
            if output_file.suffix == '.json':
                json.dump(list(results), output, cls=conversions.EnhancedJSONEncoder)
            else:
                writer = csv.writer(output)
                writer.writerows((result.psm_comment, result.crop, result.scenario, result.pec) for result in results)


def _make_runs(run_data: Iterable[Tuple[Union[Path, str], FOCUSCrop, FrozenSet[Scenario]]]) -> \
        Generator[Tuple[Union[Path, str], FOCUSCrop, Scenario], None, None]:
    for psm_file, crop, scenarios in run_data:
        for scenario in scenarios:
            yield psm_file, crop, scenario


def run_psms(run_data: Iterable[Tuple[Union[Path, str], FOCUSCrop, FrozenSet[Scenario]]],
             max_workers: int = cpu_count() - 1) -> Generator[PelmoResult, None, None]:
    """Run all given psm_files.
    When given scenarios that are not defined for some given crops, they are silently ignored for those crops only
    :param max_workers: How many worker threads to use at most
    :param run_data: Information about the Pelmo runs that should be started. Each value is a tuple containing
    the psm file, the crop to use and the defined scenarios for the runs in that order
    :return: A Generator of the results of the calculations. Makes new results available as their calculations finish.
                No particular ordering is guaranteed but the calculations are started in order of
                psm_file, then crop, then scenario"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        extract_zip(temp_dir / 'sample', Path(__file__).parent / 'data' / 'FOCUS.zip')
        with ThreadPoolExecutor(thread_name_prefix='pelmo_runner', max_workers=max_workers, initializer=_init_thread, initargs=(temp_dir,)) as executor:
            tasks = [executor.submit(single_pelmo_run, run_info, temp_dir) for run_info in _make_runs(run_data=run_data)]
            for future in as_completed(tasks):
                try:
                    yield future.result()
                except ValueError as e:
                    logging.getLogger().warning('A Pelmo run failed, excluding it from results', exc_info=e)


def find_duration(psm_file_string) -> int:
    """Given a psm file, find out how long it runs"""
    in_application = False
    max_year = 0
    for line in psm_file_string.splitlines():
        if not in_application:
            if line.startswith("<APPLICATION>"):
                in_application = True
            continue
        if in_application:
            if line.startswith("<END APPLICATION>"):
                break
            try:
                current = int(line.split()[2])
                max_year = max(max_year, current)
            except ValueError:
                pass
            except IndexError:
                pass
    if max_year <= 26:
        return 26
    elif max_year <= 46:
        return 46
    else:
        return max_year


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
    thread_dir = working_dir / current_thread().name / "FOCUS"
    if isinstance(psm_file, Path):
        run_dir = thread_dir / f'{psm_file.stem}.run'
    else:
        run_dir = thread_dir / f'{hash(psm_file)}.run'
    crop_dir = run_dir / f"{crop.focus_name.replace(' ', '_-_')}.run"
    scenario_dir = crop_dir / f"{scenario.value}_-_({scenario.name}).run"

    logger.debug('Creating run directory %s', crop_dir)
    scenario_dir.mkdir(exist_ok=True, parents=True)
    if isinstance(psm_file, Path):
        target_psm_file = scenario_dir / psm_file.name
    else:
        target_psm_file = scenario_dir / f"{hash(psm_file)}.psm"
    target_inp_file = scenario_dir / 'pelmo.inp'
    target_dat_file = scenario_dir / 'input.dat'

    logger.debug('Creating pelmo input files')
    if isinstance(psm_file, Path):
        psm_file_string = psm_file.read_text(encoding="windows-1252")
        target_psm_file.write_text(psm_file_string, encoding="windows-1252")
    else:
        psm_file_string = psm_file
        target_psm_file.write_text(psm_file, encoding="windows-1252")
        psm_file = target_psm_file

    duration = find_duration(psm_file_string)
    crop_file_name = lookup_crop_file_name(crop)
    target_inp_file.write_text(inp_file_template.render(psm_file=psm_file, crop_file_name=crop_file_name,
                                                        scenario=scenario, duration=duration),
                               encoding="windows-1252")
    target_dat_file.write_text(dat_file_template.render(),
                               encoding="windows-1252")
    shutil.copy(thread_dir / f"{scenario.name}_{crop_file_name}.crp",
                scenario_dir / f"{scenario.name}_{crop_file_name}.crp")
    logger.info('Starting PELMO run for compound: %s :: crop: %s :: scenario: %s', target_psm_file.stem,
                crop.focus_name, scenario.value)
    try:
        process = subprocess.run([str((Path(__file__).parent / 'data' / 'PELMO500.EXE').absolute())], cwd=scenario_dir,
                                 check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise ValueError(
            f"Pelmo exited prematurely with an error while calculating {psm_file.name} {crop.focus_name} "
            f"{scenario.value}. Pelmo returned exitcode {e.returncode}. The Pelmo error was {e.output}")
    logger.info('Finished PELMO run for %s, %s and %s', target_psm_file.stem, crop.focus_name, scenario.value)
    if b"F A T A L   E R R O R" in process.stdout:
        raise ValueError(
            f"Pelmo completed with error while calculating {psm_file.name} {crop.focus_name} {scenario.value}. "
            f"The Pelmo output was {process.stdout.decode(errors='backslashreplace')}")

    with psm_file.open() as psm:
        psm.readline()
        psm_comment = psm.readline()

    result = PelmoResult(psm_comment=psm_comment, scenario=scenario, crop=crop, pec=parse_pelmo_result(scenario_dir))
    if logger.level > logging.DEBUG:
        rmtree(scenario_dir)
    return result


def parse_pelmo_result(run_dir: Path) -> Dict[str, float]:
    """Parses the Pelmo output files to determine the PEC that pelmo calculated
    :param run_dir: Where Pelmo was executed and placed its result files
    :return: A list of concentrations, element 0 is the parent
    and the others are the Metabolites in the order defined by Pelmo"""
    water_file = run_dir / "WASSER.PLM"
    chem_files = run_dir.glob("CHEM*.PLM")

    m1_compartment = 20

    water_plm = WaterPLM(water_file)
    m1_water_height = [year[m1_compartment].leaching_output for year in water_plm.horizons][6:]
    pecs_per_application_period = len(m1_water_height) // 20
    mean_water_height = []
    for index in range(0, len(m1_water_height), pecs_per_application_period):
        mean_water_height.append(
            sum(m1_water_height[index:(index + pecs_per_application_period)]) / pecs_per_application_period)
    m1_water_height = mean_water_height
    results = {}
    for chem_file in chem_files:
        if chem_file.stem == "CHEM":
            compound_pec = "parent"
        else:
            compound_pec = chem_file.stem.split('_')[1]
        chem_plm = ChemPLM(chem_file)

        # for some extemely high values that don't normally appear pelmo capitulates and stops recording the values, treat these values as infinite
        m1_chem_mass = [year[m1_compartment].leaching_output if len(year) >= m1_compartment else float('inf') for year in chem_plm.horizons][6:]
        mean_chem_mass = []
        for index in range(0, len(m1_chem_mass), pecs_per_application_period):
            mean_chem_mass.append(
                sum(m1_chem_mass[index:(index + pecs_per_application_period)]) / pecs_per_application_period)
        m1_chem_mass = mean_chem_mass
        #       calculate result in kg/(cm*ha)                 convert to microgram/L
        m1_pecs = [chemical / water * 10_000 if water > 0 else 0
                   for chemical, water in zip(m1_chem_mass, m1_water_height)]
        # mass in g/ha / water in mm
        # input is in kg/ha and cm
        m1_pecs.sort()
        percentile = 0.8
        lower = int((len(m1_pecs) - 1) * percentile)
        pec_groundwater = (m1_pecs[lower] + m1_pecs[lower + 1]) / 2
        results[compound_pec] = pec_groundwater

    return results


def extract_zip(working_dir: Path, focus_zip: Path):
    """Extract the given zip file to the working directory
    """
    logger = logging.getLogger()

    logger.info(f'Extracting {focus_zip.name} to {working_dir.name}')
    with ZipFile(focus_zip) as zip_file:
        zip_file.extractall(path=working_dir)
    logger.info('Finished extracting zip')


def parse_args() -> Namespace:
    """Parse all arguments"""
    parser = ArgumentParser()
    parser.add_argument('-p', '--psm-files', type=Path, required=True,
                        help="The psm file to run. If this is a directory, run all .psm files in this directory")
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
                        type=gap.FOCUSCrop.parse, default=list(gap.FOCUSCrop),
                        help="The crops to simulate. Can be specified multiple times. "
                             "Should be listed as a two letter acronym. "
                             "The selected crops have to be present in the FOCUS zip, "
                             "the bundled zip includes all crops. Defaults to all crops.")
    parser.add_argument('-s', '--scenario', nargs='*', type=lambda x: correct_type(x, Scenario),
                        default=list(gap.Scenario),
                        help="The scenarios to simulate. Can be specified multiple times. Defaults to all scenarios. "
                             "A scenario will be calculated if it is defined both here and for the crop")
    parser.add_argument('-t', '--threads', type=int, default=cpu_count() - 1,
                        help="The maximum number of threads for Pelmo. Defaults to cpu_count - 1")
    parser.add_argument('-o', '--output', type=Path, default=Path('output.json'),
                        help="Where to write the results to. Defaults to output.json")
    parser.add_argument('--pessimistic-interception', action='store_true',
                        help='Use only the interception value of the first application')
    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger()

    jsonLogger.configure_logger_from_argparse(logger, args)
    return args


if __name__ == '__main__':
    main()
