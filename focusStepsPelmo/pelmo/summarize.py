"""A file for functions for summarizing outputs from pelmo"""
import csv
import functools
import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Tuple, Type, Union

from focusStepsPelmo.ioTypes.combination import Combination
from focusStepsPelmo.ioTypes.compound import Compound
from focusStepsPelmo.ioTypes.gap import GAP
from focusStepsPelmo.ioTypes.pelmo import PECResult, PelmoResult
from focusStepsPelmo.util.conversions import EnhancedJSONEncoder
from focusStepsPelmo.util.datastructures import correct_type


def main():
    """Main entry point for summarize script when run standalone"""
    args = parse_args()
    logger = logging.getLogger()
    logger.debug(args)
    rebuild_scattered_to_file(file=args.output, parent=args.source, input_directories=args.input_location,
                              glob_pattern=args.glob_pattern, pessimistic_interception=args.pessimistic_interception)


def parse_args() -> Namespace:
    """Parse the args for this script when run directly"""
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', default=Path('out.json'), type=Path,
                        help="Where to write the summary results")
    parser.add_argument('-s', '--source', required=True, type=Path,
                        help="The parent directory of the output files to summarize")
    parser.add_argument('-i', '--input-location', required=True, nargs='+', type=Path,
                        help="The locations of the input files")
    parser.add_argument('-g', '--glob_pattern', default="*output.json", type=str,
                        help="The glob pattern the output files conform to")
    parser.add_argument('--pessimistic-interception', action='store_true',
                        help='Use only the interception value of the first application')
    args = parser.parse_args()
    args.input_location = tuple(args.input_location)
    return args


def rebuild_scattered_to_file(file: Path, parent: Path, input_directories: Tuple[Path, ...], pessimistic_interception: bool,
                              glob_pattern: str = "*output.json" ):
    """Rebuild output in multiple locations to one location
    :param file: The final output file
    :param parent: The parent directory of the output files
    :param input_directories: Where to find the input data to connect to the output files
    :param glob_pattern: The pattern to find the output files to combine"""
    write_results_to_file(rebuild_scattered_output(parent, input_directories, glob_pattern), file, pessimistic_interception)


async def rebuild_scattered_to_file_async(file: Path, parent: Path, input_directories: Tuple[Path, ...],
                                          glob_pattern: str = "*output.json"):
    """Rebuild output in multiple locations to one location
    :param file: The final output file
    :param parent: The parent directory of the output files
    :param input_directories: Where to find the input data to connect to the output files
    :param glob_pattern: The pattern to find the output files to combine"""
    rebuild_scattered_to_file(file, parent, input_directories, glob_pattern=glob_pattern, pessimistic_interception=False)


def rebuild_output_to_file(file: Path,
                           results: Union[Path, Iterable[PelmoResult]], input_directories: Tuple[Path, ...],
                           pessimistic_interception: bool):
    """Rebuild output in one location to one location
        :param file: The final output file
        :param input_directories: Where to find the input data to connect to the output files
        :param results: The file with the pelmo results or a list of the results"""
    write_results_to_file(rebuild_output(results, input_directories), file, pessimistic_interception)


def write_results_to_file(results: Iterable[PECResult], file: Path, pessimistic_interception: bool):
    """Write the results to the output file in the output format indicated by the filename suffix of file"""
    output_format = file.suffix[1:]
    file.parent.mkdir(exist_ok=True, parents=True)
    if output_format == 'json':
        with file.with_suffix('.json').open('w') as fp:
            results = list(results)
            json.dump(results, fp, cls=EnhancedJSONEncoder)
    elif output_format == 'csv':
        with file.with_suffix('.csv').open('w', newline='') as fp:
            writer = csv.writer(fp, )
            results_iter = iter(results)
            first_result: PECResult = next(results_iter)
            writer.writerow(first_result.get_csv_headers())
            writer.writerow(first_result.to_list(pessimistic_interception))
            writer.writerows(r.to_list(pessimistic_interception) for r in results_iter)
    else:
        raise ValueError("Could not infer format, please specify explicitly")


def rebuild_scattered_output(parent: Path, input_directories: Tuple[Path, ...], glob_pattern: str = "*output.json",
                             ) -> Generator[PECResult, None, None]:
    """Rebuild the output from Pelmo together with the input files. Fetches the Pelmo result from multiple places"""
    logger = logging.getLogger()
    logger.debug("Iterating over output files %s", list(parent.rglob(glob_pattern)))
    for file in parent.rglob(glob_pattern):
        yield from rebuild_output(file, input_directories)


def rebuild_output(source: Union[Path, Iterable[PelmoResult]], input_directories: Tuple[Path, ...]
                   ) -> Generator[PECResult, None, None]:
    """Rebuild the output from Pelmo together with the input files"""
    logger = logging.getLogger()
    if isinstance(source, Path):
        with source.open() as fp:
            outputs = json.load(fp)
        outputs = [PelmoResult(**item) for item in outputs]
    else:
        outputs = source
    for output in outputs:
        input_data_hashes = json.loads(output.psm_comment)
        if 'compound' in input_data_hashes.keys() and 'gap' in input_data_hashes.keys():
            compound_hash = input_data_hashes['compound']
            gap_hash = input_data_hashes['gap']
            logger.debug({"compound_file": compound_hash, "gap_file": gap_hash})
            compound: Compound = get_obj_by_hash(h=compound_hash, file_roots=input_directories) # type: ignore - retrieving by hash would need a hash collision to yield something other than a this class
            gap: GAP = get_obj_by_hash(h=gap_hash, file_roots=input_directories) # type: ignore - retrieving by hash would need a hash collision to yield something other than a this class
        elif 'combination' in input_data_hashes.keys():
            combination_hash = input_data_hashes['combination']
            combination: Combination = get_obj_by_hash(h=combination_hash, file_roots=input_directories) # type: ignore - retrieving by hash would need a hash collision to yield something other than a this class
            compound = combination.compound
            gap = combination.gap
        else:
            raise ValueError('Could not find origin input data for psm file. '
                             'Is the comment set to the metadata generated by creator.py?')
        pecs = {}
        if compound.metabolites:
            all_metabolites = {desc.metabolite for desc in compound.metabolites}
        else:
            all_metabolites = set()
        all_metabolites = all_metabolites.union(
            {desc.metabolite for metabolite in all_metabolites for desc in metabolite.metabolites})
        for compound_name, pec in output.pec.items():
            if compound_name == "parent":
                pecs[compound.name] = pec
            else:
                for metabolite in all_metabolites:
                    if 'pelmo' in metabolite.model_specific_data.keys():
                        if compound_name.casefold() == metabolite.model_specific_data['pelmo']['position'].casefold():
                            pecs[metabolite.name] = pec
                            break
                else:  # if for wasn't completed by break
                    first_letter = compound_name[0].upper()
                    order = compound_name[1]
                    metabolite = compound.metabolites[ord(first_letter) - ord('A')].metabolite
                    if order == "2":
                        metabolite = metabolite.metabolites[0].metabolite
                    pecs[metabolite.name] = pec

        yield PECResult(compound=compound, gap=gap, scenario=output.scenario, pec=pecs)


@functools.lru_cache(maxsize=None)
def get_obj_by_hash(h: int, file_roots: Iterable[Path]) -> Union[Compound, GAP, Combination]:
    """Given a hash of an object and a file_root to search, find an object with that hash in file_root"""
    hashes = {}
    for file_root in file_roots:
        hashes.update(get_hash_obj_relation(file_root, (Compound, GAP, Combination)))
    return hashes[h]


@functools.lru_cache(maxsize=None)
def get_hash_obj_relation(directory: Path, candidate_classes: Tuple[Type, ...]) -> Dict[int, Any]:
    """Get a mapping from hashes to objects for all objects in files in directory"""
    hashes = {}
    if directory.is_dir():
        files = directory.glob('*.json')
    elif directory.exists():
        files = [directory]
    else:
        files = []
    from_file_candidates = {candidate for candidate in candidate_classes if hasattr(candidate, 'from_file')}
    json_candidates = {candidate for candidate in candidate_classes if candidate not in from_file_candidates}
    for file in files:
        for candidate in from_file_candidates:
            try:
                objs = list(candidate.from_file(file))
            except TypeError:
                continue
            for obj in objs:
                hashes[hash(obj)] = obj
            if len(objs) == 1:
                try:
                    hashes[int(file.stem)] = objs[0]
                except ValueError:
                    pass
        for candidate in json_candidates:
            with file.open() as fp:
                json_data = json.load(fp)
            # noinspection PyBroadException
            try:
                # noinspection PyTypeChecker
                obj = correct_type(json_data, candidate)
            except TypeError:
                # Failure of conversion is expected to frequently happen
                continue
            hashes[hash(obj)] = obj

    return hashes


if __name__ == '__main__':
    main()
