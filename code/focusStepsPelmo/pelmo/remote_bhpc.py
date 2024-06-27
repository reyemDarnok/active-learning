from argparse import ArgumentParser, Namespace
import logging
import os
from pathlib import Path
from shutil import rmtree
from contextlib import suppress
import shutil
import subprocess
import sys

sys.path += [str(Path(__file__).parent.parent)]

from typing import Generator, Iterable, Optional, Sequence, Tuple, TypeVar
from zipfile import ZipFile
import zipfile
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from bhpc import commands
import util.jsonLogger as jsonLogger
from util import conversions
from ioTypes.gap import FOCUSCrop, Scenario
from pelmo.creator import generate_psm_files
from pelmo.summarize import rebuild_scattered_to_file

jinja_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=select_autoescape(),
                        undefined=StrictUndefined)

T = TypeVar('T')


def split_into_batches(iterable: Iterable[T], batch_size=1) -> Generator[Generator[T], None, None]:
    """Lazily split a Sequence into a Generator of equally sized slices of the sequence. The last slice may be smaller if the sequence does not evenly divide into the batch size
    :param iterable: The iterable to split. Will be lazily evaluated
    :param batch_size: The size of a given slice. The final slice will be shorter if the length of the sequence is not divisible by batch size
    :yield: A slize of sequence of length batch size or, if it is the final slice and the length of the sequence is not divisible by batch size, smaller"""

    def batch():
        for _ in range(batch_size):
            # noinspection PyTypeChecker
            yield next(iterable)
    while True:
        yield batch()


def main():
    args = parse_args()
    logger = logging.getLogger()
    logger.debug(args)
    run_bhpc(work_dir=args.work_dir, compound_file=args.compound_file, gap_file=args.gap_file, submit=args.submit,
             output=args.output,
             crops=args.crops, scenarios=args.scenarios,
             notification_email=args.notification_email, session_timeout=args.session_timeout, run=args.run)


def run_bhpc(work_dir: Path, submit: Path, output: Path, compound_file: Path = None, gap_file: Path = None,
             combination_dir: Path = None,
             crops: Iterable[FOCUSCrop] = FOCUSCrop, scenarios: Iterable[Scenario] = Scenario,
             notification_email: Optional[str] = None, session_timeout: int = 6, run: bool = True):
    logger = logging.getLogger()
    logger.info('Starting to generate psm files')
    psm_file_data = generate_psm_files(compounds=compound_file, gaps=gap_file, combinations=combination_dir)
    crops = list(crops)
    scenarios = list(scenarios)
    machines, cores, batch_number = find_core_bhpc_configuration(crops, scenarios, compound_file, gap_file,
                                                                 compound_file)

    with suppress(FileNotFoundError):
        rmtree(submit)
    logger.info('Generating sub files for bhpc')
    make_sub_file(psm_file_data=psm_file_data, target_dir=submit,
                  crops=crops, scenarios=scenarios,
                  batch_number=batch_number)

    if run:
        logger.info('Starting Pelmo run')
        session = commands.start_submit_file(submit_folder=submit, session_name_prefix='Pelmo',
                                             submit_file_regex='pelmo\\.sub',
                                             machines=machines, cores=cores, multithreading=True,
                                             notification_email=notification_email, session_timeout=session_timeout)
        logger.info('Started Pelmo run as session %s', session)
        commands.download(session)
        rebuild_scattered_to_file(output, submit, [x for x in (gap_file, compound_file, combination_dir) if x],
                                  "psm*.d-output.json", submit)
        commands.remove(session)


def find_core_bhpc_configuration(crops: Sequence[FOCUSCrop], scenarios: Sequence[Scenario],
                                 compound_file: Optional[Path], gap_file: Optional[Path],
                                 combination_file: Optional[Path]) -> Tuple[int, int, int]:
    psm_count = 0
    if compound_file:
        if compound_file.is_dir():
            psm_count = len(list(compound_file.glob('*')))
        else:
            psm_count = 1
    if gap_file:
        if gap_file.is_dir():
            psm_count *= len(list(gap_file.glob('*')))
    if combination_file:
        if combination_file.is_dir():
            psm_count += len(list(combination_file.glob('*')))
        else:
            psm_count += 1
    logger = logging.getLogger()
    single_pelmo_instance = 15  # seconds
    crop_scenario_combinations = 0
    for crop in crops:
        crop_scenario_combinations += len(set(crop.defined_scenarios).intersection(scenarios))
    single_pelmo = single_pelmo_instance * crop_scenario_combinations  # in seconds
    target_duration = 120 * 60  # two hours
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
        machines = max(1, int(desired_core_count) // 95)
    batch_number = machines
    logger.info(f'Determined to run with {cores} cores on {machines} machines')
    return machines, cores, batch_number


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
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-r', str((script_dir / 'requirements.txt').absolute()), '--platform',
         'win32', '--upgrade', '--only-binary', ':all:', '--target', str(output.absolute())], capture_output=True,
        check=True)
    logger.debug('Getting datatypes')
    shutil.copytree(script_dir / '..' / 'ioTypes', output / 'ioTypes')
    logger.debug('Getting utils')
    shutil.copytree(script_dir / '..' / 'util', output / 'util')


def zip_directory(directory: Path, zip_name: str):
    """Zips all directory into zipName, such that if zipName is unzipped to the directory it is in directory will be restored
    :param directory: The directory to zip
    :param zip_name: The name of the resulting zip
    """
    logger = logging.getLogger()
    with ZipFile(file=directory.parent / zip_name, mode='a', compression=zipfile.ZIP_DEFLATED) as zip:
        for root, _, files in os.walk(directory):
            for file in files:
                root = Path(root)
                file = Path(file)
                combined = root / file
                pl = [x.name for x in combined.parents if x != directory.parent and x not in directory.parent.parents]
                pl.reverse()
                logger.debug("%s has %s as registered parents", combined, pl)
                name_in_archive = Path(*pl) / combined.name
                logger.debug('adding %s to zip as %s', combined.absolute(), name_in_archive)

                zip.write(root / file, name_in_archive)


def make_sub_file(psm_file_data: Iterable[str], target_dir: Path,
                  crops: Iterable[FOCUSCrop] = FOCUSCrop, scenarios: Iterable[Scenario] = Scenario,
                  batch_number: int = 1):
    '''Creates a BHPC Submit file for the Pelmo runs. WARNING: Moves the psm files to target_dir while working
    :param psm_file_data: The contents of the psm files to be included in the submit file
    :param target_dir: The directory to write the sub file to
    :param crops: The crops to run. Crop / scenario combinations that are not defined are silently skipped
    :param scenarios: The scenarios to run. Scenario / crop combinations that are not defined are silently skipped
    :param batch_number: How many batches of psm files to make, each can be run in parallel'''
    logger = logging.getLogger()

    logger.info('Setting up file system for the sub file')
    copy_common_files(output=target_dir)
    zip_common_directories(target=target_dir)

    logger.info('Making batches')
    batch_directory_names = make_batches(psm_file_data, target_dir, batch_number)

    logger.info('Writing sub file')
    sub_template = jinja_env.get_template('commit.sub.j2')
    submit_file = target_dir / "pelmo.sub"
    submit_file.write_text(sub_template.render(
        batches=batch_directory_names,
        crops=crops,
        scenarios=scenarios,
    ))
    logger.info('Finished creating files')


def make_batches(psm_file_data: Iterable[str], target_dir: Path, batch_size: int = 10000) -> Generator[str, None, None]:
    """Create the directories for the batches and fill them
    :param psm_file_data: The psm files to batch.
    :param target_dir: The parent directory for the batch directories
    :param batch_size: How many psm files to batch into one
    :return: The names of the created batches"""
    logger = logging.getLogger()
    logger.info('Splitting psm_files into batches')
    batches = split_into_batches(psm_file_data, batch_size)
    logger.info('Split psm_files into batches')
    for i, batch in enumerate(batches):
        batch_name = f"psm{i}.d"
        batch_folder = target_dir / batch_name
        batch_folder.mkdir(exist_ok=True, parents=True)
        logger.info('Adding psm files for batch %s', i)
        for file in batch:
            logger.debug('Adding %s to batch %s', file, i)
            file.rename(batch_folder / file.name)
        logger.info('Created batch %s', i)
        for psm_file in psm_file_data:
            with ZipFile(target_dir / f"{batch_name}.zip", 'w', zipfile.ZIP_DEFLATED) as zip:
                zip.writestr(str(Path(batch_name, f"{hash(psm_file)}.psm")), psm_file)
        yield batch_name


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-c', '--compound-file', required=True, type=Path,
                        help='The compound to create a psm file for. If this is a directory, create psm files for every compound file in the directory, with .json files assumed to be compound files and no recursion')
    parser.add_argument('-g', '--gap-file', required=True, type=Path,
                        help='The gap to create a psm file for. If this is a directory, create psm files for every gap file in the directory, with .json files assumed to be compound files and no recursion')
    parser.add_argument('-w', '--work-dir', default=Path.cwd() / 'pelmofiles', type=Path,
                        help='The directory in which files for Pelmo will be created. Defaults to the current directory')
    parser.add_argument('-s', '--submit', default=Path('submit'), type=Path,
                        help='Where to output the submit file and its dependencies. Defaults to "submit"')
    parser.add_argument('-o', '--output', default=Path('output'), type=Path,
                        help='Where to output the final results. Defaults to output')
    parser.add_argument('--crop', nargs='*', default=FOCUSCrop, type=FOCUSCrop.from_acronym,
                        help="Which crops to run. Defaults to all crops")
    parser.add_argument('--scenario', nargs='*', type=lambda x: conversions.str_to_enum(x, Scenario),
                        default=list(Scenario),
                        help="The scenarios to simulate. Can be specified multiple times. Defaults to all scenarios. "
                             "A scenario will be calculated if it is defined both here and for the crop")
    parser.add_argument('-r', '--run', action='store_true', default=False,
                        help="Run the created submit files on the bhpc")
    parser.add_argument('--notification-email', type=str, default=None,
                        help="The email address which will be notified if the bhpc run finishes")
    parser.add_argument('--session-timeout', type=int, default=6, help="How long should the bhpc run at most")
    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger()
    jsonLogger.configure_logger_from_argparse(logger, args)
    return args


if __name__ == '__main__':
    main()
