#!/usr/bin/env python3
import logging
from focusStepsDatatypes.pelmo import ChemPLM, PelmoResult, WaterPLM
import jsonLogger
from argparse import Namespace, ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path
import subprocess
from typing import Iterable, List
from zipfile import ZipFile
from shutil import copytree, rmtree
from concurrent.futures import Future, ThreadPoolExecutor
from threading import current_thread

from jinja2 import Environment, PackageLoader, StrictUndefined, Template, select_autoescape
from focusStepsDatatypes import helperfunctions
from focusStepsDatatypes.gap import PelmoCrop, Scenario
from focusStepsDatatypes import gap
from contextlib import suppress

jinja_env = Environment(loader=PackageLoader("psm_runner"), autoescape=select_autoescape(), undefined=StrictUndefined)

logger = logging.getLogger()

def _init_thread(working_dir: Path):
        # PELMO can't run multiple times in the same directory at the same time
    logger.debug('Starting initialisation of %s', current_thread().name)
    runner_dir = working_dir / current_thread().name
    sample_dir = working_dir / 'sample'
    copytree(sample_dir, runner_dir)
    logger.info('%s is initialised', current_thread().name)

def main():
    args = parse_args()
    logger.debug(args)
    with suppress(FileNotFoundError): rmtree(args.working_dir)
    if args.focus_dir.is_dir():
        copytree(args.focus_dir, args.working_dir / 'sample' )
    else:
        extract_zip(args.working_dir / 'sample', args.focus_dir)
    files = args.psm_files.glob('*.psm') if args.psm_files.is_dir() else [args.psm_files]
    results = run_psms(files, args.working_dir, args.crop, args.scenario, args.pelmo_exe, args.threads)
    logger.info(results)

            

def run_psms(psm_files: Iterable[Path], working_dir: Path, crops: Iterable[PelmoCrop] = PelmoCrop, scenarios: Iterable[Scenario] = Scenario, pelmo_exe: Path = Path('C:/FOCUS_PELMO.664/PELMO500.exe'), max_workers: int = cpu_count() - 1) -> List[float]:
    pool = ThreadPoolExecutor(max_workers=max_workers, initializer=_init_thread, initargs=(working_dir,))
    futures: List[Future] = []
    for psm_file in psm_files:
        inp_file_template = jinja_env.get_template('pelmo.inp.j2')
        dat_file_template = jinja_env.get_template('input.dat')
        for crop in crops:
            for scenario in set(crop.defined_scenarios).intersection(scenarios):
                futures.append(pool.submit(single_pelmo_run, pelmo_exe, psm_file, working_dir, inp_file_template, dat_file_template, crop, scenario))
    result = []
    for f in futures:
        try:
            result.append(f.result())
        except ValueError as e:
            logger.warn(e)
    pool.shutdown()
    return result

def single_pelmo_run(pelmo_exe: Path, psm_file: Path, working_dir: Path, inp_file_template: Template, dat_file_template: Template, crop: PelmoCrop, scenario: Scenario) -> float:
    scenario_dirs = working_dir / current_thread().name
    scenario_dir = scenario_dirs / scenario.value
    run_dir = scenario_dir / f'{psm_file.stem}.run'
    crop_dir = run_dir / crop.display_name

    logger.debug('Creating run directory %s', crop_dir)
    crop_dir.mkdir(exist_ok=True, parents=True)

    target_psm_file = crop_dir / psm_file.name
    target_inp_file = crop_dir / 'pelmo.inp'
    target_dat_file = crop_dir / 'input.dat'

    logger.info('Creating pelmo input files')
    target_psm_file.write_text(psm_file.read_text())
    target_inp_file.write_text(inp_file_template.render(psm_file = psm_file, crop=crop, scenario=scenario))
    target_dat_file.write_text(dat_file_template.render())

    logger.info('Starting PELMO run for compound: %s :: crop: %s :: scenario: %s', target_psm_file.stem, crop.display_name, scenario.value)
    try:
        process = subprocess.run([str(pelmo_exe.absolute())], cwd=crop_dir, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Pelmo exited prematurely with an error while calculating {psm_file.name} {crop.display_name} {scenario.value}. Pelmo returned exitcode {e.returncode}. The Pelmo error was {e.output}")
    logger.info('Finished PELMO run for %s, %s and %s', target_psm_file.stem, crop.display_name, scenario.value)
    if b"F A T A L   E R R O R" in process.stdout:
        raise ValueError(f"Pelmo completed with error while calculating {psm_file.name} {crop.display_name} {scenario.value}. The Pelmo output was {process.stdout.decode(errors='backslashreplace')}")

    return parse_pelmo_result(psm_file)





def parse_pelmo_result(psm_file: Path, target_compartment = 21) -> List[PelmoResult]:
    run_dir = psm_file.parent
    water_file = run_dir / "WASSER.PLM"
    chem_files = run_dir.glob("CHEM*.PLM")

    water_plm = WaterPLM(water_file)
    results = []
    for chem_file in chem_files:
        chem_plm = ChemPLM(chem_file)
        p = PelmoResult(water_plm, chem_plm)
        results.append(p)


    return results





def extract_zip(working_dir: Path, focus_zip: Path):
    logger.info(f'Extracting {focus_zip.name} to {working_dir.name}')
    with ZipFile(focus_zip) as zip:
        zip.extractall(path=working_dir)
    logger.info('Finished extracting zip')



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-p', '--psm-files', type=Path, required=True, help="The psm file to run. If this is a directory, run all .psm files in this directory")
    parser.add_argument('-w', '--working-dir', type=Path, default=Path.cwd(), help="The directory to use as a root working directory. Will be filled with expanded zips and defaults to the current working directory")
    parser.add_argument('-f', '--focus-dir', type=Path, default=Path(__file__).parent / 'Focus.zip', help="The PELMO FOCUS directory to use. If a zip, will be unpacked first. Defaults to a bundled zip.")
    parser.add_argument('-e', '--pelmo-exe', type=Path, default=Path('C:/FOCUS_PELMO.664') / 'PELMO500.exe', help="The PELMO executable to use for running. Defaults to the default PELMO installation. This should point to the CLI EXE, usually named PELMO500.EXE NOT to the GUI EXE usually named wpelmo.exe.")
    parser.add_argument('-c', '--crop', nargs='*', type=gap.PelmoCrop.from_acronym, default=list(gap.PelmoCrop), help="The crops to simulate. Can be specified multiple times. Should be listed as a two letter acronym. The selected crops have to be present in the FOCUS zip, the bundled zip includes all crops. Defaults to all crops.")
    parser.add_argument('-s', '--scenario', nargs='*', type=lambda x: helperfunctions.str_to_enum(x, Scenario), default=list(gap.Scenario), help="The scenarios to simulate. Can be specified multiple times. Defaults to all scenarios. A scenario will be calculated if it is defined both here and for the crop")
    parser.add_argument('-t', '--threads', type=int, default=cpu_count() - 1, help="The maximum number of threads for Pelmo. Defaults to cpu_count - 1")
    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    jsonLogger.configure_logger_from_argparse(logger, args)

if __name__ == '__main__':
    main()