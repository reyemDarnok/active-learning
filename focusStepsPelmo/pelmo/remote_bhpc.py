import itertools
import logging
import os
import shutil
import zipfile
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from multiprocessing import cpu_count
from pathlib import Path
from shutil import rmtree
from typing import Generator, Iterable, Optional, TypeVar, List
from zipfile import ZipFile

from jinja2 import Environment, StrictUndefined, select_autoescape, PackageLoader

from focusStepsPelmo.bhpc import commands
from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.ioTypes.compound import Compound
from focusStepsPelmo.ioTypes.gap import FOCUSCrop, Scenario, GAP
from focusStepsPelmo.pelmo.creator import generate_psm_files
from focusStepsPelmo.pelmo.summarize import rebuild_scattered_to_file
from focusStepsPelmo.util import jsonLogger
from focusStepsPelmo.util.datastructures import correct_type
from focusStepsPelmo.util.iterable_helper import repeat_infinite, count_up

jinja_env = Environment(loader=PackageLoader('focusStepsPelmo.pelmo'),
                        autoescape=select_autoescape(), undefined=StrictUndefined)

T = TypeVar('T')


def split_into_batches(iterable: Iterable[T], batch_size=1, fillvalue: T = None) -> Generator[List[T], None, None]:
    """Lazily split a Sequence into a Generator of equally sized slices of the sequence.
    The last slice may be smaller if the sequence does not evenly divide into the batch size
    :param iterable: The iterable to split. Will be lazily evaluated
    :param batch_size: The size of a given slice.
    :param fillvalue: fill the final batch to size with this item
    :return: A generator for lists of size batch_size from iterable
    >>> def count_up(limit: int) -> Generator[int, None, None]:
    ...     current = 0
    ...     while current < limit:
    ...         yield current
    ...         current += 1
    >>> list(split_into_batches(count_up(10), 2))
    [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    >>> list(split_into_batches(count_up(10), 4))
    [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, None, None)]
    """
    iterator = iterable.__iter__()
    return itertools.zip_longest(*[iterator] * batch_size, fillvalue=fillvalue)


def main():
    args = parse_args()
    logger = logging.getLogger()
    logger.debug(args)
    run_bhpc(compound_file=args.compound_file, gap_file=args.gap_file, submit=args.submit,
             output=args.output,
             crops=args.crop, scenarios=args.scenario,
             notification_email=args.notification_email, session_timeout=args.session_timeout, run=args.run)


def run_bhpc(submit: Path, output: Path, compound_file: Path = None, gap_file: Path = None,
             combination_dir: Path = None,
             crops: Iterable[FOCUSCrop] = FOCUSCrop, scenarios: Iterable[Scenario] = Scenario,
             notification_email: Optional[str] = None, session_timeout: int = 6, run: bool = True):
    logger = logging.getLogger()
    logger.info('Starting to generate psm files')
    psm_file_data = generate_psm_files(compounds=Compound.from_path(compound_file) if compound_file else None,
                                       gaps=GAP.from_path(gap_file) if gap_file else None,
                                       combinations=Combination.from_path(combination_dir) if combination_dir else None)
    crops = list(crops)
    scenarios = list(scenarios)

    with suppress(FileNotFoundError):
        rmtree(submit)
    logger.info('Generating sub files for bhpc')
    batch_number = make_sub_file(psm_file_data=psm_file_data, target_dir=submit,
                                 crops=crops, scenarios=scenarios,
                                 batch_size=1000)
    if run:
        logger.info('Starting Pelmo run')
        if batch_number > 10:
            cores = 96
        elif batch_number > 2:
            cores = 16
        else:
            cores = 8
        session = commands.start_submit_file(submit_folder=submit, session_name_prefix='Pelmo',
                                             submit_file_regex='pelmo\\.sub',
                                             machines=max(1, batch_number // 10), cores=cores, multithreading=True,
                                             notification_email=notification_email, session_timeout=session_timeout)
        logger.info('Started Pelmo run as session %s', session)
        commands.download(session)
        rebuild_scattered_to_file(file=output, parent=submit,
                                  input_directories=tuple(x for x in (gap_file, compound_file, combination_dir) if x),
                                  glob_pattern="psm*.d-output.json")
        commands.remove(session)


def zip_common_directories(target: Path):
    """Zip directories in output to common.zip
    so that when common.zip is unpacked the directories are where they started
    :param target: The directory to search for directories to zip"""
    logger = logging.getLogger()
    for directory in os.listdir(target):
        directory = Path(target / directory)
        if directory.is_dir():
            logger.debug('Adding %s to zip', directory)
            zip_directory(Path(directory), 'common.zip')


def copy_common_files(output: Path):
    """Copies the files common to all runs to output
    :param output: Where to copy the files to
    """
    logger = logging.getLogger()
    output.mkdir(exist_ok=True, parents=True)
    script_dir = Path(__file__).parent.parent
    logger.debug('Copying script')
    shutil.copytree(script_dir, output / 'focusStepsPelmo')
    logger.debug('Copying pythonwrapper')
    shutil.copy(script_dir / 'pelmo' / 'pythonwrapper.bat', output / 'pythonwrapper.bat')
    logger.debug('Copying requirements.txt')
    shutil.copy(script_dir / 'pelmo' / 'requirements.txt', output / 'requirements.txt')


def zip_directory(directory: Path, zip_name: str):
    """Zips all directory into zipName,
    such that if zipName is unzipped to the directory it is in directory will be restored
    :param directory: The directory to zip
    :param zip_name: The name of the resulting zip
    """
    logger = logging.getLogger()
    with ZipFile(file=directory.parent / zip_name, mode='a', compression=zipfile.ZIP_DEFLATED) as zip_file:
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

                zip_file.write(root / file, name_in_archive)


def make_sub_file(psm_file_data: Iterable[str], target_dir: Path,
                  crops: Iterable[FOCUSCrop] = FOCUSCrop, scenarios: Iterable[Scenario] = Scenario,
                  batch_size: int = 1000) -> int:
    """Creates a BHPC Submit file for the Pelmo runs. WARNING: Moves the psm files to target_dir while working
    :param psm_file_data: The contents of the psm files to be included in the submit file
    :param target_dir: The directory to write the sub file to
    :param crops: The crops to run. Crop / scenario combinations that are not defined are silently skipped
    :param scenarios: The scenarios to run. Scenario / crop combinations that are not defined are silently skipped
    :param batch_size: How large a batch of psm files should be, each can be run in parallel
    :return: The number of jobs in the sub file"""
    logger = logging.getLogger()

    logger.info('Setting up file system for the sub file')
    copy_common_files(output=target_dir)
    zip_common_directories(target=target_dir)

    logger.info('Making batches')
    batch_directory_names = make_batches(psm_file_data, target_dir, batch_size)
    batch_directory_names = list(batch_directory_names)
    logger.info('Writing sub file')
    sub_template = jinja_env.get_template('commit.sub.j2')
    submit_file = target_dir / "pelmo.sub"
    submit_file.write_text(sub_template.render(
        batches=batch_directory_names,
        crops=crops,
        scenarios=scenarios,
    ))
    logger.info('Finished creating files')
    return len(batch_directory_names)


def make_batches(psm_file_data: Iterable[str], target_dir: Path, batch_size: int = 1000) -> Generator[str, None, None]:
    """Create the directories for the batches and fill them
    :param psm_file_data: The psm files to batch.
    :param target_dir: The parent directory for the batch directories
    :param batch_size: How many psm files to batch into one
    :return: The names of the created batches"""
    logger = logging.getLogger()
    logger.info('Splitting psm_files into batches')
    batches = split_into_batches(psm_file_data, batch_size)
    logger.info('Split psm_files into batches')
    pool = ThreadPoolExecutor(max_workers=cpu_count() - 1)
    logger.info('Initialized Thread Pool')
    yield from pool.map(make_batch,
                        count_up(),
                        batches,
                        repeat_infinite(target_dir))
    logger.info('Registered all batch creation functions')
    pool.shutdown()


def make_batch(index: int, batch: Iterable[str], target_dir: Path) -> str:
    logger = logging.getLogger()
    batch_name = f"psm{index}.d"
    logger.info('Adding psm files for batch %s', index)
    with ZipFile(target_dir / f"{batch_name}.zip", 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for psm_file in batch:
            if psm_file is not None:
                zip_file.writestr(str(Path(batch_name, f"{hash(psm_file)}.psm")), psm_file)
    logger.info('Created batch %s', index)
    return batch_name


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-c', '--compound-file', default=None, type=Path,
                        help='The compound to create a psm file for. '
                             'If this is a directory, create psm files for every compound file in the directory, '
                             'with .json files assumed to be compound files and no recursion')
    parser.add_argument('-g', '--gap-file', default=None, type=Path,
                        help='The gap to create a psm file for. If this is a directory, '
                             'create psm files for every gap file in the directory, '
                             'with .json files assumed to be compound files and no recursion')
    parser.add_argument('--combined', default=None, type=Path,
                        help="Combinations of gaps and compounds. If it is a directory, parse every .json file in"
                             "that directory")
    parser.add_argument('-s', '--submit', default=Path('submit'), type=Path,
                        help='Where to output the submit file and its dependencies. Defaults to "submit"')
    parser.add_argument('-o', '--output', default=Path('output'), type=Path,
                        help='Where to output the final results. Defaults to output')
    parser.add_argument('--crop', nargs='*', default=FOCUSCrop, type=FOCUSCrop.parse,
                        help="Which crops to run. Defaults to all crops")
    parser.add_argument('--scenario', nargs='*', type=lambda x: correct_type(x, Scenario),
                        default=list(Scenario),
                        help="The scenarios to simulate. Can be specified multiple times. Defaults to all scenarios. "
                             "A scenario will be calculated if it is defined both here and for the crop")
    parser.add_argument('-r', '--run', action='store_true', default=False,
                        help="Run the created submit files on the bhpc")
    parser.add_argument('--notification-email', type=str, default=None,
                        help="The email address which will be notified if the bhpc run finishes")
    parser.add_argument('--session-timeout', type=int, default=6,
                        help="How long should the bhpc run at most")
    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger()
    jsonLogger.configure_logger_from_argparse(logger, args)
    return args


if __name__ == '__main__':
    main()
