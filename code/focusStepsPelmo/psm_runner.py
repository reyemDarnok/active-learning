#!/usr/bin/env python3
from argparse import Namespace, ArgumentParser
from pathlib import Path
import subprocess
from typing import Iterable, List
from zipfile import ZipFile
from shutil import copytree, rmtree
from concurrent.futures import Future, ThreadPoolExecutor
from threading import current_thread

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from focusStepsDatatypes.gap import Crop, PelmoCrop
from focusStepsDatatypes import gap
from contextlib import suppress
import logging
logging.basicConfig(level=logging.DEBUG, format='%(module)-12s :: %(asctime)s :: %(levelname)-8s :: %(threadName)-22s :: %(funcName)-12s :: %(message)s')

jinja_env = Environment(loader=PackageLoader("main"), autoescape=select_autoescape(), undefined=StrictUndefined)

def _init_thread(working_dir: Path):
        # PELMO can't run multiple times in the same directory at the same time
    logging.debug('Starting initialisation of %s', current_thread().name)
    runner_dir = working_dir / current_thread().name
    sample_dir = working_dir / 'sample'
    copytree(sample_dir, runner_dir)
    logging.info('%s is initialised', current_thread().name)

def main():
    args = parse_args()
    logging.debug(args)
    with suppress(FileNotFoundError): rmtree(args.working_dir)
    if args.focus_dir.is_dir():
        copytree(args.focus_dir, args.working_dir / 'sample' )
    else:
        extract_zip(args.working_dir / 'sample', args.focus_dir)
    files = args.psm_files.glob('*.psm') if args.psm_files.is_dir() else [args.psm_files]
    run_psms(files, args.working_dir, args.crop, args.pelmo_exe)

            

def run_psms(psm_files: Iterable[Path], working_dir: Path, crops: Iterable[PelmoCrop], pelmo_exe: Path = Path('C:/FOCUS_PELMO.664/PELMO500.exe')):
    executor = ThreadPoolExecutor(max_workers=4, initializer=_init_thread, initargs=(working_dir,))
    for psm_file in psm_files:
        logging.debug('Submitting %s to ThreadPool', psm_file.name)
        executor.submit(_run_psm, psm_file, working_dir, crops, pelmo_exe)
    executor.shutdown(wait=True)

def _run_psm(psm_file: Path, working_dir: Path, crops: Iterable[PelmoCrop], pelmo_exe: Path) -> bool:
    scenario_dirs = working_dir /current_thread().name
    inp_file_template = jinja_env.get_template('pelmo.inp.j2')
    dat_file_template = jinja_env.get_template('input.dat')
    for crop in crops:
        for scenario in crop.defined_scenarios:
            scenario_dir = scenario_dirs / scenario.value
            run_dir = scenario_dir / f'{psm_file.stem}.run'
            crop_dir = run_dir / crop.display_name

            logging.debug('Creating run directory %s', crop_dir)
            crop_dir.mkdir(exist_ok=True, parents=True)

            target_psm_file = crop_dir / psm_file.name
            target_inp_file = crop_dir / 'pelmo.inp'
            target_dat_file = crop_dir / 'input.dat'

            logging.info('Creating pelmo input files')
            target_psm_file.write_text(psm_file.read_text())
            target_inp_file.write_text(inp_file_template.render(psm_file = psm_file, crop=crop, scenario=scenario))
            target_dat_file.write_text(dat_file_template.render())

            logging.info('Starting PELMO run for %s, %s and %s', target_psm_file.stem, crop.display_name, scenario.value)
            subprocess.run([str(pelmo_exe.absolute())], cwd=crop_dir, check=True) #, stdout=subprocess.DEVNULL)
            logging.info('Finished PELMO run for %s, %s and %s', target_psm_file.stem, crop.display_name, scenario.value)
    return True

def extract_zip(working_dir: Path, focus_zip: Path):
    logging.info(f'Extracting {focus_zip.name} to {working_dir.name}')
    with ZipFile(focus_zip) as zip:
        zip.extractall(path=working_dir)
    logging.info('Finished extracting zip')



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-p', '--psm-files', type=Path, required=True, help="The psm file to run. If this is a directory, run all .psm files in this directory")
    parser.add_argument('-w', '--working-dir', type=Path, default=Path.cwd(), help="The directory to use as a root working directory. Will be filled with expanded zips and defaults to the current working directory")
    parser.add_argument('-f', '--focus-dir', type=Path, default=Path(__file__).parent / 'Focus.zip', help="The PELMO FOCUS directory to use. If a zip, will be unpacked first. Defaults to a bundled zip.")
    parser.add_argument('-e', '--pelmo-exe', type=Path, default=Path('C:/FOCUS_PELMO.664') / 'PELMO500.exe', help="The PELMO executable to use for running. Defaults to the default PELMO installation. This should point to the CLI EXE, usually named PELMO500.EXE NOT to the GUI EXE usually named wpelmo.exe.")
    parser.add_argument('-c', '--crop', nargs='+', type=gap.PelmoCrop.from_acronym, default=list(gap.PelmoCrop), help="The crops to simulate. Can be specified multiple times. Should be listed as a two letter acronym. The selected crops have to be present in the FOCUS zip, the bundled zip includes all crops. Defaults to all crops.")
    return parser.parse_args()


if __name__ == '__main__':
    main()