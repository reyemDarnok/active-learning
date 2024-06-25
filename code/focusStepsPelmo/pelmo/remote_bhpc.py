from argparse import ArgumentParser, Namespace
import json
import logging
import os
from pathlib import Path
from shutil import rmtree
from contextlib import suppress
import shutil
import subprocess
import sys

sys.path += [str(Path(__file__).parent.parent)]

from typing import Generator, Iterable, Optional, Sequence, TypeVar
from zipfile import ZipFile
import zipfile
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from bhpc import commands
import util.jsonLogger as jsonLogger
from util import conversions
from util.conversions import EnhancedJSONEncoder
from ioTypes.gap import FOCUSCrop, Scenario
from pelmo.creator import generate_psm_files
from pelmo.summarize import rebuild_scattered_output



jinja_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=select_autoescape(), undefined=StrictUndefined)

T = TypeVar('T')
def split_into_batches(sequence: Sequence[T], batchsize=1) -> Generator[Sequence[T], None, None]:
    """Lazily split a Sequence into a Generator of equally sized slices of the sequence. The last slice may be smaller if the sequence does not evenly divide into the batch size
    :param sequence: The sequnce to split. Will be lazily evaluated
    :param batchsize: The size of a given slice. The final slice will be shorter if the lenght of the sequence is not divisible by batchsize
    :yield: A slize of sequence of length batchsize or, if it is the final slice and the length of the sequence is not divisible by batchsize, smaller"""
    length = len(sequence)
    for index in range(0, length, batchsize):
        yield sequence[index:min(index + batchsize, length)]

def split_into_n_batches(sequence: Sequence[T], batchnumber=1) -> Generator[Sequence[T], None, None]:
    length = len(sequence)
    batchsize = length // batchnumber
    if length % batchnumber != 0:
        batchsize += 1
    yield from split_into_batches(sequence, batchsize)

def main():
    args = parse_args()
    logger = logging.getLogger()
    logger.debug(args)
    run_bhpc(work_dir=args.work_dir, compound_file=args.compound_file,gap_file=args.gap_file, submit=args.submit, output=args.output, output_format=args.output_format,
             crops=args.crops, scenarios=args.scenarios, 
             notificationemail=args.notification_email, session_timeout=args.session_timeout, run=args.run)

def run_bhpc(work_dir: Path, submit: Path, output: Path, compound_file: Path = None, gap_file: Path = None, combination_dir: Path = None, output_format: str = 'json',
             crops: Iterable[FOCUSCrop] = FOCUSCrop, scenarios: Iterable[Scenario] = Scenario, 
             notificationemail: Optional[str] = None, session_timeout: int = 6, run: bool = True):
    logger = logging.getLogger()
    
    if output_format is None and output:
            output_format = output.suffix[1:]
    logger.info('Starting to genearte psm files')
    psm_dir: Path = work_dir / 'psm'
    psm_count = generate_psm_files(output_dir=psm_dir, compound_file=compound_file, gap_file=gap_file, combination_dir=combination_dir)

    single_pelmo_instance = 15 # seconds
    crop_scenario_combinations = 0
    crops = list(crops)
    scenarios = list(scenarios)
    for crop in crops:
        crop_scenario_combinations += len(set(crop.defined_scenarios).intersection(scenarios))
    single_pelmo = single_pelmo_instance * crop_scenario_combinations # in seconds
    target_duration = 120 * 60 # two hours
    single_core_duration = single_pelmo * psm_count
    desired_core_count = single_core_duration / target_duration
    machines = 1
    if desired_core_count < 3:
        cores = 2
    elif desired_core_count < 7:
        cores = 4
    elif desired_core_count < 15:
        cores = 8
    elif desired_core_count < 47:
        cores = 16
    else:
        cores = 96
        machines = max(1, desired_core_count // 95)
    batchnumber = machines
    logger.info(f'Determined to run with {cores} cores on {machines} machines')

    with suppress(FileNotFoundError): rmtree(submit)
    logger.info('Generating sub files for bhpc')
    psm_files = list(psm_dir.glob('*.psm'))
    logger.debug('Collected psm files')
    logger.debug(psm_files)
    make_sub_file(psm_files=psm_files, target_dir=submit, 
                  crops=crops, scenarios=scenarios, 
                  batchnumber=batchnumber, output_format=output_format)

    if run:
        logger.info('Starting Pelmo run')
        session = commands.start_submit_file(submit_folder=submit, session_name_prefix='Pelmo', submit_file_regex='pelmo\\.sub',
                               machines=machines, cores=cores, multithreading=True,
                               notificationemail=notificationemail, session_timeout=session_timeout)
        logger.info('Started Pelmo run as session %s', session)
        commands.download(session)
        results = rebuild_scattered_output(submit, "psm*.d-output.json", psm_root=submit)
        with output.with_suffix(f'.{output_format}').open('w') as output_file:
            if output_format == "json":
                results = list(results)
                json.dump(results, output_file, cls=EnhancedJSONEncoder)
            elif output_format == "csv":
                for result in conversions.flatten_to_csv(results):
                    output_file.write(result)
            else:
                raise ValueError(f"Invalid output format {output_format}")
        commands.remove(session)


def zip_common_directories(target: Path):
    """Zip directories in output to common.zip so that when common.zip is unpacked the directories are where they started
    :param target: The directory to search for directories to zip"""
    logger = logging.getLogger()
    for directory in os.listdir(target):
        directory = Path(target / directory)
        logger.debug('Considering to add %s to zip', directory)
        if directory.is_dir():
            logger.debug('Adding %s to zip', directory)
            zip_directory(Path(directory), 'common.zip')

def copy_common_files(output: Path):
    """Copies the files common to all runs to output
    :param output: Where to copy the files to
    """
    logger = logging.getLogger()
    output.mkdir(exist_ok=True, parents=True)
    script_dir = Path(__file__).parent
    logger.debug('Copying script')
    shutil.copytree(script_dir, output / 'pelmo')
    logger.debug('Copying pythonwrapper')
    shutil.copy(script_dir / 'pythonwrapper.bat', output / 'pythonwrapper.bat')     
    logger.debug('Getting requirements')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str((script_dir / 'requirements.txt').absolute()),  '--platform', 'win32',  '--upgrade', '--only-binary', ':all:', '--target', str(output.absolute())], capture_output=True, check=True)
    logger.debug('Getting datatypes')
    shutil.copytree(script_dir / '..' / 'ioTypes', output / 'ioTypes')
    logger.debug('Getting utils')
    shutil.copytree(script_dir / '..' / 'util', output / 'util')


def zip_directory(directory: Path, zip_name:str, mode: str='a'):
    """Zips all directory into zipName, such that if zipName is unzipped to the directory it is in directory will be restored
    :param directory: The directory to zip
    :param zip_name: The name of the resulting zip
    :param mode: The mode to use when opening the zip file. Reasonable values are "w" (overwrite old file) and "a" (append to old file)"""
    logger = logging.getLogger()
    with ZipFile(directory.parent / zip_name, mode, zipfile.ZIP_DEFLATED) as zip:
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


def make_sub_file(psm_files: Iterable[Path], target_dir: Path, 
                  crops: Iterable[FOCUSCrop] = FOCUSCrop, scenarios: Iterable[Scenario] = Scenario, 
                  batchnumber: int = 1, output_format: str = 'csv'):
    '''Creates a BHPC Submit file for the Pelmo runs. WARNING: Moves the psm files to target_dir while working
    :param psm_files: The files to run in Pelmo. WARNING: Will be moved to target_dir
    :param target_dir: The directory to write the sub file to
    :param crops: The crops to run. Crop / scenario combinations that are not defined are silently skipped
    :param scenarios: The scenarios to run. Scenario / crop combinations that are not defined are silently skipped
    :param batchsize: How many psm files per bhpc job'''
    logger = logging.getLogger()

    logger.info('Setting up file system for the sub file')
    copy_common_files(output=target_dir)
    zip_common_directories(target=target_dir)

    logger.info('Making batches')
    batchdirs = make_batches(psm_files, target_dir, batchnumber)
    
    logger.info('Writing sub file')
    sub_template = jinja_env.get_template('commit.sub.j2')
    subfile = target_dir / "pelmo.sub"
    subfile.write_text(sub_template.render(
        batches=batchdirs,
        crops= crops,
        scenarios=scenarios,
        output_format=output_format
    ))
    logger.info('Finished creating files')

def make_batches(psm_files: Iterable[Path], target_dir: Path, batchnumber: int):
    """Create the directories for the batches and fill them
    :param psm_files: The psm files to batch. WARNING: They will be moved, not copied to target_dir
    :param target_dir: The parent directory for the batch directories
    :param batchsize: How many psm files to a batch"""
    logger = logging.getLogger()
    logger.info('Splitting psm_files into batches')
    batches = list(split_into_n_batches(psm_files, batchnumber))
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
        zip_directory(batch_folder, f"{batchname}.zip")
    batchdirs = [f"psm{i}.d" for i in range(len(batches))]
    return batchdirs

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add
    parser.add_argument('-c', '--compound-file', required=True, type=Path, help='The compound to create a psm file for. If this is a directory, create psm files for every compound file in the directory, with .json files assumed to be compound files and no recursion')
    parser.add_argument('-g', '--gap-file', required=True, type=Path, help='The gap to create a psm file for. If this is a directory, create psm files for every gap file in the directory, with .json files assumed to be compound files and no recursion')
    parser.add_argument('-w', '--work-dir', default=Path.cwd() / 'pelmofiles', type=Path, help='The directory in which files for Pelmo will be created. Defaults to the current directory')
    parser.add_argument('-s', '--submit', default=Path('submit'), type=Path, help='Where to output the submit file and its dependencies. Defaults to "submit"')
    parser.add_argument('-o', '--output', default=Path('output'), type=Path, help='Where to output the final results. Defaults to output')
    parser.add_argument(      '--crop', nargs='*', default=FOCUSCrop, type=FOCUSCrop.from_acronym, help="Which crops to run. Defaults to all crops")
    parser.add_argument(      '--scenario', nargs='*', type=lambda x: conversions.str_to_enum(x, Scenario), default=list(Scenario), help="The scenarios to simulate. Can be specified multiple times. Defaults to all scenarios. A scenario will be calculated if it is defined both here and for the crop")
    parser.add_argument('-r', '--run', action='store_true', default=False, help="Run the created submit files on the bhpc")
    parser.add_argument('--notification-email', type=str, default=None, help="The email address which will be notified if the bhpc run finishes")
    parser.add_argument('--session-timeout', type=int, default=6, help="How long should the bhpc run at most")
    parser.add_argument('--output-format', type=str.lower, choices=("json", "csv"), default=None, help="The output format. Defaults to guessing from the file name")
    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger()
    jsonLogger.configure_logger_from_argparse(logger, args)
    return args

if __name__ == '__main__':
    main()

