#!/usr/bin/env python3
import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Generator, Iterable, Type, TypeVar, Union

from jinja2 import Environment, FileSystemLoader, select_autoescape, StrictUndefined

from psm_file import PsmFile
from ..ioTypes.combination import Combination
from ..ioTypes.compound import Compound
from ..ioTypes.gap import GAP
from ..util import jsonLogger as jsonLogger

jinja_env = Environment(loader=FileSystemLoader(
    [Path(__file__).parent / "templates", Path(__file__).parent / "templates" / "psm-fragments"]),
                        autoescape=select_autoescape(), undefined=StrictUndefined)


def main():
    args = parse_args()
    write_psm_files(output_dir=args.output_dir, compounds=args.compound_file, gaps=args.gap_file,
                    combinations=args.combination_file)


def write_psm_files(output_dir: Path,
                    compounds: Iterable[Union[Path, Compound]] = None,
                    gaps: Iterable[Union[Path, Compound]] = None,
                    combinations: Iterable[Union[Path, Compound]] = None) -> int:
    if compounds:
        compounds = load_or_use(compounds, Compound)
    if gaps:
        gaps = load_or_use(gaps, GAP)
    if combinations:
        combinations = load_or_use(combinations, Combination)
    total = 0
    for psm_file in generate_psm_files(compounds=compounds, gaps=gaps, combinations=combinations):
        total += 1
        (output_dir / f"{hash(psm_file)}.psm").write_text(psm_file)
    return total


T = TypeVar('T')


def load_or_use(it: Iterable[Union[Path, T]], t: Type[T]) -> Generator[T, None, None]:
    for element in it:
        if isinstance(element, t):
            yield element
        else:
            yield from load_class(element, t)


def load_class(source: Path, t: Type[T]) -> Generator[T, None, None]:
    if source.suffix == '.json':
        with source.open() as file:
            json_content = json.load(file)
        if isinstance(json_content, list):
            for element in json_content:
                yield t(**element)
        else:
            yield t(**json_content)
    elif source.suffix == 'xlsx':
        yield from t.from_excel(source)


def generate_psm_files(compounds: Iterable[Compound] = None, gaps: Iterable[GAP] = None,
                       combinations: Iterable[Combination] = None) -> Generator[str, None, None]:
    assert not (bool(compounds) ^ bool(gaps)), "Either both or neither of compound file have to be specified"
    if combinations:
        for combination in combinations:
            comment = json.dumps({"combination": hash(Combination)})
            yield _generate_psm_contents(compound=combination.compound, gap=combination.gap, comment=comment)
    if compounds and gaps:
        gaps = list(gaps)
        for compound in compounds:
            for gap in gaps:
                comment = json.dumps({"compound": hash(compound), "gap": hash(gap)})
                yield _generate_psm_contents(compound, gap, comment)


def _generate_psm_contents(compound: Compound, gap: GAP, comment: str) -> str:
    """For a given compound and gap file, generate the matching psm files 
    :param gap: The gap file to use when generating psm file
    :param compound: The compound file to use when generating psm file
    :param comment: The comment in the resulting psm file
    :return: The contents of the psm file"""

    psm_file = PsmFile.from_input(compound=compound, gap=gap)
    psm_template = jinja_env.get_template('general.psm.j2')
    psm_file.comment = comment
    # noinspection PyProtectedMember
    return psm_template.render(**psm_file._asdict())


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-c', '--compound-file', default=None, type=Path,
                        help='The compound to create a psm file for. If this is a directory, create psm files for '
                             'every compound file in the directory, with .json files assumed to be compound files and '
                             'no recursion')
    parser.add_argument('--combination_file', default=None, type=Path,
                        help='A file of combinations of gap and compounds. '
                             'If this is a directory, read all files in that directory')
    parser.add_argument('-g', '--gap-file', default=None, type=Path,
                        help='The gap to create a psm file for. If this is a directory, create psm files for every '
                             'gap file in the directory, with .json files assumed to be compound files and no '
                             'recursion')
    parser.add_argument('-o', '--output-dir', default=Path('output'), type=Path,
                        help='The directory for output files. '
                             'The files will be named {COMPOUND_FILE}-{GAP_FILE}-{MATURATION}-{DAY}.psm. '
                             'Defaults to a folder named output')
    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger()
    jsonLogger.configure_logger_from_argparse(logger, args)
    return args


if __name__ == '__main__':
    main()
