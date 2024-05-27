#!/usr/bin/env python3
from argparse import Namespace, ArgumentParser
from pathlib import Path
import subprocess
from typing import List
from zipfile import ZipFile
from shutil import copytree
from concurrent.futures import ThreadPoolExecutor

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from focusStepsDatatypes.gap import Crop, PelmoCrop
from focusStepsDatatypes import gap
from contextlib import suppress
import logging
logging.basicConfig(level=logging.DEBUG)

jinja_env = Environment(loader=PackageLoader("main"), autoescape=select_autoescape(), undefined=StrictUndefined)


def main():
    args = parse_args()
    with suppress(FileNotFoundError): args.working_dir.unlink()
    extract_zip(args.working_dir, args.focus_zip)
    pool = ThreadPoolExecutor(max_workers=4)
    run_psm(args.psm_file, args.working_dir / 'FOCUS', args.example_run, args.crop, args.pelmo_exe, pool)
    pool.shutdown(wait=True)

def run_psm(psm_file: Path, focus_dir: Path, sample_run: Path, crops: List[PelmoCrop], pelmo_exe: Path, thread_pool: ThreadPoolExecutor):
    sample_run = focus_dir / sample_run
    target_run = focus_dir / f'{psm_file.stem}.run'
    copytree(sample_run, target_run)
    inp_file_template = jinja_env.get_template('pelmo.inp.j2')
    for crop in crops:
        logging.debug(f'Looping for {crop.value}')
        crop_run_dir = target_run / f'{crop.value}.run'
        logging.debug(f'crop_run_dir={crop_run_dir}')
        for scenario_dir in crop_run_dir.glob('*.run'):
                logging.debug(f'Executing in {scenario_dir}')
            #def psm_function(): PELMO can't run multiple times in the same directory at the same time
                target_psm_file = scenario_dir / psm_file.name
                target_inp_file = scenario_dir / 'pelmo.inp'
                target_psm_file.write_text(psm_file.read_text())
                target_inp_file.write_text(inp_file_template.render(psm_file = psm_file))
                logging.info(f'Starting PELMO run for {target_psm_file}')
                logging.debug(f'command={[str(pelmo_exe.absolute())]}')
                logging.debug(f'cwd={scenario_dir}')
                subprocess.run([str(pelmo_exe.absolute())], cwd=scenario_dir, check=True)
                logging.info(f'Finished PELMO run for {target_psm_file}')
            #thread_pool.submit(psm_function)

def extract_zip(working_dir: Path, focus_zip: Path):
    with ZipFile(focus_zip) as zip:
        zip.extractall(path=working_dir)



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-p', '--psm-file', type=Path, required=True, help="The psm file to run. If this is a directory, run all .psm files in this directory")
    parser.add_argument('-w', '--working-dir', type=Path, default=Path.cwd(), help="The directory to use as a root working directory. Will be filled with expanded zips and defaults to the current working directory")
    parser.add_argument('-f', '--focus-zip', type=Path, default=Path(__file__).parent / 'Focus.zip', help="The PELMO FOCUS zip to use. Defaults to a bundled zip.")
    parser.add_argument(      '--example-run', type=Path, default=Path('sample.run'), help='The example folder to copy for new psm runs. Has to present in the FOCUS zip. Defaults to sample.run')
    parser.add_argument('-e', '--pelmo-exe', type=Path, default=Path('C:/FOCUS_PELMO.664') / 'PELMO500.exe', help="The PELMO executable to use for running. Defaults to the default PELMO installation. This should point to the CLI EXE, usually named PELMO500.EXE NOT to the GUI EXE usually named wpelmo.exe.")
    parser.add_argument('-c', '--crop', nargs='+', type=gap.PelmoCrop.from_acronym, default=list(gap.PelmoCrop), help="The crops to simulate. Can be specified multiple times. Should be listed as a two letter acronym. The selected crops have to be present in the FOCUS zip, the bundled zip includes all crops. Defaults to all crops.")
    return parser.parse_args()


if __name__ == '__main__':
    main()