from argparse import ArgumentParser, Namespace
from itertools import islice
import logging
import os
from pathlib import Path
from shutil import rmtree
from contextlib import suppress
import shutil
import subprocess
import sys
from typing import Iterable, List
from zipfile import ZipFile
import zipfile
import pip

from jinja2 import Environment, FileSystemLoader, PackageLoader, StrictUndefined, select_autoescape

import bhpc
import jsonLogger
from focusStepsDatatypes import helperfunctions
from focusStepsDatatypes.gap import PelmoCrop, Scenario
from psm_creator import generate_psm_files


jinja_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=select_autoescape(), undefined=StrictUndefined)

def split_into_batches(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def main():
    args = parse_args()
    logger = logging.getLogger()
    logger.debug(args)
    logger.info('Deleting old artefacts')
    with suppress(FileNotFoundError): rmtree(args.work_dir)
    psm_dir: Path = args.work_dir / 'psm'
    logger.info('Starting to genearte psm files')
    generate_psm_files(output_dir=psm_dir, compound_file=args.compound_file, gap_file=args.gap_file)
    logger.info('Setting up file system for the sub file')
    with suppress(FileNotFoundError): rmtree(args.output)
    args.output.mkdir(exist_ok=True, parents=True)
    script_dir = Path(__file__).parent
    logger.debug('Copying psm_runner.py')
    shutil.copy(script_dir / 'psm_runner.py', args.output / 'psm_runner.py')
    logger.debug('Copying focus zip')
    shutil.copy(script_dir / args.focus_zip, args.output / 'FOCUS.zip')
    logger.debug('Copying pelmo')
    shutil.copy(script_dir / args.pelmo_exe, args.output / 'PELMO500.exe')
    logger.debug('Copying datatypes')
    shutil.copytree(script_dir / 'focusStepsDatatypes', args.output / 'focusStepsDatatypes')
    logger.debug('Copying templates')
    shutil.copytree(script_dir / 'templates', args.output / 'templates')
    logger.debug('Copying JsonLogger')
    shutil.copy(script_dir / 'jsonLogger.py', args.output / 'jsonLogger.py')   
    logger.debug('Copying pythonwrapper')
    shutil.copy(script_dir / 'pythonwrapper.bat', args.output / 'pythonwrapper.bat')     
    logger.debug('Getting jinja2')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str((script_dir / 'requirements.txt').absolute()),  '--platform', 'win32',  '--upgrade', '--only-binary', ':all:', '--target', str(args.output.absolute())])
    logger.info('Generating sub files for bhpc')
    psm_files = list(psm_dir.glob('*.psm'))
    logger.debug('Collected psm files')
    logger.debug(psm_files)
    make_sub_file(psm_files=psm_files, target_dir=args.output, crops=args.crop, scenarios=args.scenario)
    for directory in os.listdir(args.output):
        directory = Path( args.output / directory)
        logger.debug('Considering to add %s to zip', directory)
        if directory.is_dir() and not directory.name.endswith('.d'):
            logger.debug('Adding %s to zip', directory)
            zip_folders(Path(directory), 'common.zip')
    if args.run:
        session = bhpc.start_submit_file(submit_folder=args.output, session_name_prefix='Pelmo', submit_file_regex='pelmo\\.sub',
                               machines=args.count, cores=args.cores, multithreading=args.multithreading,
                               notificationemail=args.notification_email, session_timeout=args.session_timeout)
        logger.info('Started Pelmo run as session %s', session)

def zip_folders(directory: Path, zipName:str):
    logger = logging.getLogger()
    with ZipFile(directory.parent / zipName, 'a', zipfile.ZIP_DEFLATED) as zip:
        for root, _, files in os.walk(directory):
            for file in files:
                root = Path(root)
                file = Path(file)
                combined = root / file
                pl = [x.name for x in combined.parents if x != directory.parent and x not in directory.parent.parents]
                pl.reverse()
                logger.debug("%s has %s as registered parents", combined, pl)
                arcname = Path(*pl) / combined.name
                logger.debug('adding %s to zip as %s', combined.absolute(), arcname)

                zip.write(root / file, arcname)


def make_sub_file(psm_files: Iterable[Path], target_dir: Path, crops: PelmoCrop = PelmoCrop, scenarios: Scenario = Scenario, batchsize: int = 1):
    '''Creates a BHPC Submit file for the Pelmo runs. WARNING: Moves the psm files to target_dir while working'''
    logger = logging.getLogger()
    sub_template = jinja_env.get_template('commit.sub.j2')
    logger.info('Splitting psm_files into batches')
    batches = list(split_into_batches(psm_files, batchsize))
    logger.info('Split psm_files into batches')
    logger.debug(batches)
    for i, batch in enumerate(batches):
        batchname = f"psm{i}.d"
        batch_folder = target_dir / batchname
        batch_folder.mkdir(exist_ok=True, parents=True)
        logger.info('Adding psm files for batch %s', i)
        for file in batch:
            logger.debug('Adding %s to batch %s', file, i)
            file.rename(batch_folder / file.name)
        logger.info('Created batch %s', i)
        zip_folders(batch_folder, f"{batchname}.zip")
    subfile = target_dir / "pelmo.sub"
    batchdirs = [f"psm{i}.d" for i in range(len(batches))]
    logger.info('Writing sub file')
    subfile.write_text(sub_template.render(
        batches=batchdirs,
        crops= crops,
        scenarios=scenarios
    ))
    logger.info('Finished creating files')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-c', '--compound-file', required=True, type=Path, help='The compound to create a psm file for. If this is a directory, create psm files for every compound file in the directory, with .json files assumed to be compound files and no recursion')
    parser.add_argument('-g', '--gap-file', required=True, type=Path, help='The gap to create a psm file for. If this is a directory, create psm files for every gap file in the directory, with .json files assumed to be compound files and no recursion')
    parser.add_argument('-w', '--work-dir', default=Path.cwd() / 'pelmofiles', type=Path, help='The directory in which files for Pelmo will be created. Defaults to the current directory')
    parser.add_argument('-o', '--output', default=Path('output'), type=Path, help='Where to output the submit file and its dependencies. Defaults to "output"')
    parser.add_argument(      '--crop', nargs='*', default=list(PelmoCrop), type=PelmoCrop.from_acronym, help="Which crops to run. Defaults to all crops")
    parser.add_argument('-s', '--scenario', nargs='*', type=lambda x: helperfunctions.str_to_enum(x, Scenario), default=list(Scenario), help="The scenarios to simulate. Can be specified multiple times. Defaults to all scenarios. A scenario will be calculated if it is defined both here and for the crop")
    parser.add_argument('-e', '--pelmo-exe', type=Path, default=Path('C:/FOCUS_PELMO.664') / 'PELMO500.exe', help="The PELMO executable to use for running. Defaults to the default PELMO installation. This should point to the CLI EXE, usually named PELMO500.EXE NOT to the GUI EXE usually named wpelmo.exe.")
    parser.add_argument('-f', '--focus-zip', type=Path, default=Path('FOCUS.zip', help="The Focus data to use. Unzips it if it is a zip. Defaults to a bundled zip"))
    parser.add_argument('-r', '--run', action='store_true', default=False, help="Run the created submit files on the bhpc")
    parser.add_argument('--count', type=int, default=1, help="How many machines to use on the bhpc")
    parser.add_argument('--cores', type=int, choices=(2,4,8,16,96), default=8, help="How many cores per machine to use. One core per machine is always overhead, so larger machines are more efficient")
    parser.add_argument('--multithreading', action='store_true', default=True, help="Use multithreading")
    parser.add_argument('--notification-email', type=str, default=None, help="The email address which will be notified if the bhpc run finishes")
    parser.add_argument('--session-timeout', type=int, default=6, help="How long should the bhpc run at most")
    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger()
    jsonLogger.configure_logger_from_argparse(logger, args)
    return args

if __name__ == '__main__':
    main()

