
import csv
from argparse import ArgumentParser, Namespace
from dataclasses import replace
import json
import logging
from os import cpu_count
from pathlib import Path
from typing import Generator, Iterable, Optional, Sequence, Tuple, Union
import sys
sys.path += [str(Path(__file__).parent.parent)]
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
    span_params = {"bbch": args.bbch, "rate": args.rate, 
                    "dt50": args.dt50, "koc": args.koc, "freundlich": args.freundlich, "plant_uptake": args.plant_uptake}
    crops = args.crop
    scenarios = args.scenario
    if args.input_file:
        with args.input_file.open() as fp:
            if args.format == 'json':
                file_span_params: dict = json.load(fp)
            else:
                rows = csv.reader(fp)
                file_span_params = {row[0]: row[0:] for row in rows}
            if not crops:
                crops = file_span_params.pop('crop', FOCUSCrop)
            if not scenarios:
                scenarios = file_span_params.pop('scenario', Scenario)
            span_params = {**file_span_params, **span_params}   

    with args.template_gap.open() as fp:
        template_gap = GAP(**json.load(fp))
    with args.template_compound.open() as fp:
        template_compound = Compound(**json.load(fp))
    span_to_dir(template_gap=template_gap, template_compound=template_compound, compound_dir=args.work_dir / "compounds", gap_dir=args.work_dir / "gaps",
                **span_params)
    if args.run == 'bhpc':
        run_bhpc(work_dir=args.work_dir / 'remote', compound_file=args.work_dir / 'compounds', gap_files=args.work_dir / 'gaps',
                 submit=args.work_dir / 'submit', output=args.output, output_format=args.output_format, crops=crops, scenarios=scenarios,
                 batchsize=args.batchsize, cores=args.cores, machines=args.machines, notificationemail=args.notificationemail, session_timeout=args.session_timeout, run=True)
    elif args.run == 'local':
        run_local(work_dir=args.work_dir / 'local', compound_files=args.work_dir / 'compounds', gap_files=args.work_dir / 'gaps', 
                  output_file=args.output, output_format=args.output_format, crops=crops, scenarios=scenarios, threads=args.threads)

    

def span_to_dir(template_gap: GAP, template_compound: Compound, compound_dir: Path, gap_dir: Optional[Path] = None,
         bbch: Iterable[int] = None, rate: Iterable[float] = None,
         dt50: Iterable[float] = None, koc: Iterable[float] = None, freundlich: Iterable[float] = None, plant_uptake: Iterable[float] = None) -> None:
    if gap_dir is None:
        gap_dir = compound_dir
    gap_dir.mkdir(exist_ok=True, parents=True)
    compound_dir.mkdir(exist_ok=True, parents=True)
    for gap in span_gap(template_gap, bbch, rate):
        with (gap_dir / f"gap-{hash(gap)}.json").open('w') as fp:
            json.dump(gap, fp, cls=EnhancedJSONEncoder)
    for compound in span_compounds(template_compound, dt50, koc, freundlich, plant_uptake):
        with (compound_dir / f"compound-{hash(compound)}.json").open('w') as fp:
            json.dump(compound, fp, cls=EnhancedJSONEncoder)

def span(template_gap: GAP, template_compound: Compound,
         bbch: Iterable[int] = None, rate: Iterable[float] = None,
         dt50: Iterable[float] = None, koc: Iterable[float] = None, freundlich: Iterable[float] = None, plant_uptake: Iterable[float] = None) -> Generator[Tuple[GAP, Compound], None, None]:
    for gap in span_gap(template_gap, bbch, rate):
        for compound in span_compounds(template_compound, dt50, koc, freundlich, plant_uptake):
            yield (gap, compound)

def span_gap(template_gaps: Union[GAP, Iterable[GAP]], bbch: Optional[Iterable[int]], rate: Optional[Iterable[float]]) -> Generator[GAP, None, None]:
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
    for gap in gaps:
        for bbch in bbchs:
            new_timing = replace(gap.application.timing, bbch_state=bbch)
            new_application = replace(gap.application, timing=new_timing)
            yield replace(gap, application=new_application)

def span_rate(gaps: Iterable[GAP], rates: Iterable[float]) -> Generator[GAP, None, None]:
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
        template_compounds = span_freundlich(template_compounds, koc)
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
    parser.add_argument('-c', '--template-compound', type=Path, required=True, help="The compound to use as a template for unchanging parameters when scanning")
    parser.add_argument('-g', '--template-gap', type=Path, required=True, help="The gap to use as a template for unchanging parameters when scanning")
    parser.add_argument('-w', '--work-dir', type=Path, required=True, help="A directory to use as scratch space")
    parser.add_argument('-o', '--output', type=Path, required=True, help="Where to write the results to")
    parser.add_argument(      '--output-format', type=str, choices=('json', 'csv'), default='json', help="Which output format to use")
    parser.add_argument(      '--input-format', type=str, choices=('csv', 'json'), default='json', help="The format of the input file. CSV Files need to have the commas for the parameter ranges escaped ")
    parser.add_argument('-i', '--input-file', type=Path, help="The input file for the scanning parameters")

    parser.add_argument(      '--crop', nargs='*', type=FOCUSCrop.from_acronym, default=list(FOCUSCrop), help="The crops to simulate. Can be specified multiple times. Should be listed as a two letter acronym. The selected crops have to be present in the FOCUS zip, the bundled zip includes all crops. Defaults to all crops.")
    parser.add_argument(      '--scenario', nargs='*', type=lambda x: conversions.str_to_enum(x, Scenario), default=list(Scenario), help="The scenarios to simulate. Can be specified multiple times. Defaults to all scenarios. A scenario will be calculated if it is defined both here and for the crop")
    parser.add_argument(      '--bbch', nargs='*', default=None, help="The bbch values to scan")
    parser.add_argument(      '--rate', nargs='*', default=None, help="The application rate values to scan")
    parser.add_argument(      '--dt50', nargs='*', default=None, help="The dt50 values to scan")
    parser.add_argument(      '--koc', nargs='*', default=None, help="The koc values to scan")
    parser.add_argument(      '--freundlich', nargs='*', default=None, help="The freundlich values to scan")
    parser.add_argument(      '--plant_uptake', nargs='*', default=None, help="The plant uptake values to scan")
    
    run_subparsers = parser.add_subparsers(dest="run", help="Where to run Pelmo. The script will only generate files but not run anything if this is not specified")
    local_parser = run_subparsers.add_parser("local", help="Run Pelmo locally")
    local_parser.add_argument('-t', '--threads', type=int, default=cpu_count() - 1, help="The maximum number of threads for Pelmo. Defaults to cpu_count - 1")
    bhpc_parser = run_subparsers.add_parser("bhpc", help="Run Pelmo on the bhpc")
    bhpc_parser.add_argument('--count', type=int, default=1, help="How many machines to use on the bhpc")
    bhpc_parser.add_argument('--cores', type=int, choices=(2,4,8,16,96), default=2, help="How many cores per machine to use. One core per machine is always overhead, so larger machines are more efficient")
    bhpc_parser.add_argument('--notification-email', type=str, default=None, help="The email address which will be notified if the bhpc run finishes")
    bhpc_parser.add_argument('--session-timeout', type=int, default=6, help="How long should the bhpc run at most")
    bhpc_parser.add_argument('--batchsize', type=int, default=100, help="How many psm files to batch together into one bhpc job") 

    jsonLogger.add_log_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger()

    jsonLogger.configure_logger_from_argparse(logger, args)
    return args

if __name__ == '__main__':
    main()