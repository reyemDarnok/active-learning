
from collections import UserDict, UserList
from contextlib import suppress
import csv
from argparse import ArgumentParser, Namespace
from dataclasses import replace
import json
import logging
import math
from os import cpu_count
import os
from pathlib import Path
import random
from shutil import rmtree
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple, Union
import sys
sys.path += [str(Path(__file__).parent.parent)]
from ioTypes.combination import Combination
from util.datastructures import HashableDict
from pelmo.remote_bhpc import run_bhpc
from pelmo.local import run_local
from util import conversions, jsonLogger
from util.conversions import EnhancedJSONEncoder
from ioTypes.compound import Compound
from ioTypes.gap import GAP, FOCUSCrop, Scenario, Timing

def main():
    args = parse_args()
    logger = logging.getLogger()
    logger.debug(args)
    compound_dir = args.work_dir / 'compounds'
    gap_dir = args.work_dir / 'gaps'
    combination_dir = args.work_dir / 'combination'
    crops = args.crop
    scenarios = args.scenario
    with args.input_file.open() as input_file:
        if args.input_format == 'json':
            create_samples_in_dirs(definition=json.load(input_file), output_dir=combination_dir, sample_size = args.sample_size)
        elif args.input_format == 'csv':
            with args.template_gap.open() as gap_file:
                template_gap = GAP(**json.load(gap_file))
            with args.template_compound.open() as compound_file:
                template_compound = Compound(**json.load(compound_file))
            rows = csv.reader(input_file)
            file_span_params = {row[0]: [x.strip() for x in row[1:]] for row in rows}
            span_to_dir(template_gap=template_gap, template_compound=template_compound, compound_dir=compound_dir, gap_dir=gap_dir,
                **file_span_params)
        else:
            raise ValueError("Cannot infer input format, please specify explictly")
    if not crops:
        crops = file_span_params.pop('crop', FOCUSCrop)
    if not scenarios:
        scenarios = file_span_params.pop('scenario', Scenario)

    

    if args.run == 'bhpc':
        run_bhpc(work_dir=args.work_dir / 'remote', compound_file=compound_dir, gap_file=gap_dir, combination_dir=combination_dir,
                 submit=args.work_dir / 'submit', output=args.output, output_format=args.output_format, crops=crops, scenarios=scenarios,
                 notificationemail=args.notification_email, session_timeout=args.session_timeout, run=True)
    elif args.run == 'local':
        run_local(work_dir=args.work_dir / 'local', compound_files=compound_dir, gap_files=gap_dir, combination_dir=combination_dir,
                  output_file=args.output, output_format=args.output_format, crops=crops, scenarios=scenarios, threads=args.threads)

    
def create_samples_in_dirs(definition: Dict, output_dir: Path, sample_size: int):
    with suppress(FileNotFoundError): rmtree(output_dir)
    output_dir.mkdir(parents=True)
    samples = create_samples(definition)
    for _ in range(sample_size):
        combination = next(samples)
        with (output_dir / f"{hash(combination)}.json").open('w') as fp:
            json.dump(combination, fp, cls=EnhancedJSONEncoder)

def create_samples(definition: Dict) -> Generator[Combination,None, None]:
    while True:
        yield Combination(**create_dict_sample(definition))

def create_any_sample(definition: Any) -> Any:
    if isinstance(definition, (dict, UserDict)):
        if 'type' in definition.keys() and 'parameters' in definition.keys():
            return evaluate_template(definition)
        return create_dict_sample(definition)
    elif isinstance(definition, (list, UserList)):
        return create_list_sample(definition)
    else:
        return definition

def create_dict_sample(definition: Dict) -> Dict:
    return {key: create_any_sample(value) for key, value in definition.items()}
        
def create_list_sample(definition: List) -> List:
    return [create_any_sample(value) for value in definition]

def evaluate_template(definition: Dict) -> Any:
    types = {
        'choices': choices_template,
        'steps': steps_template,
        'random': random_template,
        'copies': copies_template,
        'dict_copies': dict_copies_template,
    }
    return types[definition['type']](**definition['parameters'])

def random_template(lower_bound: float, upper_bound: float, log_random: bool) -> float:
    if log_random:
        lower_bound = math.log(lower_bound)
        upper_bound = math.log(upper_bound)
    result = random.uniform(lower_bound, upper_bound)
    if log_random:
        result = math.exp(result)
    return result

def copies_template(minimum: int, maximum: int, value: Any) -> List[Any]:
    return [create_any_sample(value) for _ in range(random.randint(minimum, maximum))]

def dict_copies_template(minimum: int, maximum: int, key: Any, value: Any) -> List[Any]:
    return {create_any_sample(key): create_any_sample(value) for _ in range(random.randint(minimum, maximum))}

def choices_template(options: List) -> Any:
    return create_any_sample(random.choice(options))

def steps_template(start: int, stop: int, step: int = 1, scale_factor: float = 1) -> float:
    return random.randrange(start, stop, step) * scale_factor

def span_to_dir(template_gap: GAP, template_compound: Compound, compound_dir: Path, gap_dir: Optional[Path] = None,
         bbch: Iterable[int] = None, rate: Iterable[float] = None,
         dt50: Iterable[float] = None, koc: Iterable[float] = None, freundlich: Iterable[float] = None, plant_uptake: Iterable[float] = None) -> None:
    """Creates compund and gap jsons for a parameter matrix
    :param template_gap: The gap to use as a template for parameters that are not in the matrix
    :param template_compound: The compund to use as a template for parameters that are not in the matrix
    :param compound_dir: The directory to write the resulting compound files to
    :param gap_dir: The directory to write the resulting gap files to. Defaults to compound_dir if not set
    :param bbch: The BBCH values in the matrix
    :param rate: The application rate values in the matrix
    :param dt50: The DT50 values in the matrix
    :param koc: The koc values in the matrix
    :param freundlich: The freundlich values in the matrix
    :param plant_uptake: The plant uptake values in the matrix"""
    if gap_dir is None:
        gap_dir = compound_dir
    with suppress(FileNotFoundError): rmtree(gap_dir)
    with suppress(FileNotFoundError): rmtree(compound_dir)
    gap_dir.mkdir(parents=True)
    compound_dir.mkdir(parents=True)
    for gap in span_gap(template_gap, bbch, rate):
        with (gap_dir / f"gap-{hash(gap)}.json").open('w') as fp:
            json.dump(gap, fp, cls=EnhancedJSONEncoder)
    for compound in span_compounds(template_compound, dt50, koc, freundlich, plant_uptake):
        with (compound_dir / f"compound-{hash(compound)}.json").open('w') as fp:
            json.dump(compound, fp, cls=EnhancedJSONEncoder)

def span(template_gap: GAP, template_compound: Compound,
         bbch: Iterable[int] = None, rate: Iterable[float] = None,
         dt50: Iterable[float] = None, koc: Iterable[float] = None, freundlich: Iterable[float] = None, plant_uptake: Iterable[float] = None) -> Generator[Tuple[GAP, Compound], None, None]:
    """Creates compound and gap combinations from a matrix
    :param template_gap: The gap to use as a template for parameters that are not in the matrix
    :param template_compound: The compund to use as a template for parameters that are not in the matrix
    :param bbch: The BBCH values in the matrix
    :param rate: The application rate values in the matrix
    :param dt50: The DT50 values in the matrix
    :param koc: The koc values in the matrix
    :param freundlich: The freundlich values in the matrix
    :param plant_uptake: The plant uptake values in the matrix
    :return: A generator that layily creates compound/gap combinations"""
    for gap in span_gap(template_gap, bbch, rate):
        for compound in span_compounds(template_compound, dt50, koc, freundlich, plant_uptake):
            yield (gap, compound)

def span_gap(template_gaps: Union[GAP, Iterable[GAP]], bbch: Optional[Iterable[int]], rate: Optional[Iterable[float]]) -> Generator[GAP, None, None]:
    """Creates gaps from a template and a matrix
    :param template_gaps: The gaps to use as templates. If an iterable, the matrix will be applied to each element
    :param bbch: The BBCH values in the matrix
    :param rate: The application rate values in the matrix
    :return: A Generator that lazily creates gaps from the matrix"""
    # Do nothing if template_gaps is iterable, make it a single item list if its a single gap
    try:
        _ = iter(template_gaps)
    except TypeError:
        template_gaps = [template_gaps]
    logger = logging.getLogger()
    if bbch is not None:
        logger.info('Spanning over bbch values')
        template_gaps = span_bbch(template_gaps, bbch)
    if rate is not None:
        logger.info('Spanning over rate values')
        template_gaps = span_rate(template_gaps, rate)
    return template_gaps

def span_bbch(gaps: Iterable[GAP], bbchs: Iterable[int]) -> Generator[GAP, None, None]:
    """Creates gaps with different bbchs
    :param gaps: The gaps to change the bbchs in
    :param bbchs: The range of bbch to use
    :return: A generator for every gap/bbch combination"""
    for gap in gaps:
        for bbch in bbchs:
            new_timing = replace(gap.application.timing, bbch_state=bbch)
            new_application = replace(gap.application, timing=new_timing)
            yield replace(gap, application=new_application)

def span_rate(gaps: Iterable[GAP], rates: Iterable[float]) -> Generator[GAP, None, None]:
    """Creates gaps with different application rates
    :param gaps: The gaps to change the application rate in
    :param bbchs: The range of application rates to use
    :return: A generator for every gap/application rate combination"""
    for gap in gaps:
        for rate in rates:
            new_application = replace(gap.application, rate=rate)
            yield replace(gap, application=new_application)

def span_compounds(template_compounds: Union[Compound, Iterable[Compound]], dt50: Optional[Iterable[float]] = None, koc: Optional[Iterable[float]] = None, freundlich: Optional[Iterable[float]] = None, plant_uptake: Optional[Iterable[float]] = None) -> Generator[Compound, None, None]:
    # Do nothing if template_compounds is iterable, make it a single item list if its a single compound
    logger = logging.getLogger()
    try:
        _ = iter(template_compounds)
    except TypeError:
        template_compounds = [template_compounds]
    if dt50 is not None:
        logger.info('Spanning over dt50 values')
        template_compounds = span_dt50(template_compounds, dt50)
    if koc is not None:
        logger.info('Spanning over koc values')
        template_compounds = span_koc(template_compounds, koc)
    if freundlich is not None:
        logger.info('Spanning over freundlich values')
        template_compounds = span_freundlich(template_compounds, freundlich)
    if plant_uptake is not None:
        logger.info('Spanning over plant uptake values')
        template_compounds = span_plant_uptake(template_compounds, plant_uptake)
    return template_compounds

def span_dt50(compounds: Iterable[Compound], dt50s: Iterable[float]) -> Generator[Compound, None, None]:
    for compound in compounds:
        for dt50 in dt50s:
            new_degradation = replace(compound.degradation, system=dt50)
            yield replace(compound, degradation = new_degradation)

def span_koc(compounds: Iterable[Compound], kocs: Iterable[float]) -> Generator[Compound, None, None]:
    for compound in compounds:
        for koc in kocs:
            new_sorption = replace(compound.sorption, koc=koc)
            yield replace(compound, sorption = new_sorption)

def span_freundlich(compounds: Iterable[Compound], freundlichs: Iterable[float]) -> Generator[Compound, None, None]:
    for compound in compounds:
        for freundlich in freundlichs:
            new_sorption = replace(compound.sorption, freundlich=freundlich)
            yield replace(compound, sorption = new_sorption)


def span_plant_uptake(compounds: Iterable[Compound], plant_uptakes: Iterable[float]) -> Generator[Compound, None, None]:
    for compound in compounds:
        for plant_uptake in plant_uptakes:
            yield replace(compound, plant_uptake=plant_uptake)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-c', '--template-compound', type=Path, default=None, help="The compound to use as a template for unchanging parameters when scanning")
    parser.add_argument('-g', '--template-gap', type=Path, default=None, help="The gap to use as a template for unchanging parameters when scanning")
    parser.add_argument('-w', '--work-dir', type=Path, default=Path('scan') / 'work', help="A directory to use as scratch space")
    parser.add_argument('-o', '--output', type=Path, default='output.csv', help="Where to write the results to")
    parser.add_argument(      '--output-format', type=str, choices=('json', 'csv'), default=None, help="Which output format to use. Defaults to guessing from the filename")
    parser.add_argument(      '--input-format', type=str, choices=('csv', 'json'), default=None, help="The format of the input file. Defaults to guessing from the filename. CSV Files start every line with a parameter name and continue it with its possible values ")
    parser.add_argument('-i', '--input-file', type=Path, required=True, help="The input file for the scanning parameters")
    parser.add_argument('-s', '--sample-size', type=int, default=1000, help="If given an json input, how many random samples to take")
    parser.add_argument(      '--crop', nargs='*', type=FOCUSCrop.from_acronym, default=list(FOCUSCrop), help="The crops to simulate. Can be specified multiple times. Should be listed as a two letter acronym. The selected crops have to be present in the FOCUS zip, the bundled zip includes all crops. Defaults to all crops.")
    parser.add_argument(      '--scenario', nargs='*', type=lambda x: conversions.str_to_enum(x, Scenario), default=list(Scenario), help="The scenarios to simulate. Can be specified multiple times. Defaults to all scenarios. A scenario will be calculated if it is defined both here and for the crop")
    
    
    run_subparsers = parser.add_subparsers(dest="run", help="Where to run Pelmo. The script will only generate files but not run anything if this is not specified")
    local_parser = run_subparsers.add_parser("local", help="Run Pelmo locally")
    local_parser.add_argument('-t', '--threads', type=int, default=cpu_count() - 1, help="The maximum number of threads for Pelmo. Defaults to cpu_count - 1")
    bhpc_parser = run_subparsers.add_parser("bhpc", help="Run Pelmo on the bhpc")
    bhpc_parser.add_argument('--notification-email', type=str, default=None, help="The email address which will be notified if the bhpc run finishes")
    bhpc_parser.add_argument('--session-timeout', type=int, default=6, help="How long should the bhpc run at most")

    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    if args.input_format == None:
        args.input_format = args.input_file.suffix[1:]

    assert args.input_format != 'csv' or (args.template_compound and args.template_gap), "CSV input requires gap and compound templates" 
    logger = logging.getLogger()
    jsonLogger.configure_logger_from_argparse(logger, args)
    return args

if __name__ == '__main__':
    main()