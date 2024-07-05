import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Dict

import pandas

from focusStepsPelmo.util.datastructures import TypeCorrecting


@dataclass(frozen=True)
class Degradation(TypeCorrecting):
    """General Degradation information"""
    system: float
    """Total System DT50"""
    soil: float
    """DT50 in soil"""
    surfaceWater: float
    """DT50 in water"""
    sediment: float
    """DT50 in sediment"""


@dataclass(frozen=True)
class Sorption(TypeCorrecting):
    """Information about the sorption behavior of a compound. Steps12 uses the koc, Pelmo uses all values"""
    koc: float
    freundlich: float


@dataclass(frozen=True)
class Volatility(TypeCorrecting):
    water_solubility: float
    vaporization_pressure: float
    reference_temperature: float


@dataclass(frozen=True)
class MetaboliteDescription(TypeCorrecting):
    formation_fraction: float
    metabolite: 'Compound'


@dataclass(frozen=True)
class Compound(TypeCorrecting):
    """A Compound definition"""
    molarMass: float
    """molar mass in g/mol"""
    volatility: Volatility
    sorption: Sorption
    """A sorption behaviour"""
    degradation: Degradation
    """Degradation behaviours"""
    plant_uptake: float = 0
    """Fraction of plant uptake"""
    name: str = field(hash=False, default="Unknown Name")  # str hash is not stable
    model_specific_data: Dict = field(compare=False, hash=False, default_factory=dict)
    metabolites: Optional[Tuple[MetaboliteDescription, ...]] = field(default_factory=tuple)
    """The compounds metabolites"""

    def metabolite_description_by_name(self, name: str) -> Optional[MetaboliteDescription]:
        if self.metabolites is not None:
            for met_des in self.metabolites:
                if met_des.metabolite.name == name:
                    return met_des
        return None

    @staticmethod
    def from_excel(excel_file: Path) -> List['Compound']:
        compounds = pandas.read_excel(io=excel_file, sheet_name="Compound Properties")
        compounds['Pelmo Position'].fillna('No Position', inplace=True)
        metabolite_relationships = pandas.read_excel(io=excel_file, sheet_name="Metabolite Relationships")
        compound_list = [
            Compound(name=row['Name'], molarMass=row['Molar Mass'],
                     volatility=Volatility(water_solubility=row['Water Solubility'],
                                           vaporization_pressure=row['Vaporization Pressure'],
                                           reference_temperature=row['Temperature']),
                     sorption=Sorption(koc=row['Koc'], freundlich=row['Freundlich']),
                     plant_uptake=row['Plant Uptake'],
                     degradation=Degradation(system=row['DT50 System'],
                                             soil=row['DT50 Soil'],
                                             sediment=row['DT50 Sediment'],
                                             surfaceWater=row['DT50 Surface Water']),
                     model_specific_data={'pelmo': {'position': row['Pelmo Position']
                     if row['Pelmo Position'] != 'No Position'
                     else None}})
            for _, row in compounds.iterrows()
        ]
        parents = [compound for compound in compound_list if
                   compound.name not in metabolite_relationships['Metabolite'].values]
        for _, row in metabolite_relationships.iterrows():
            parent: Compound = next(filter(lambda c: c.name == row['Parent'], compound_list))
            metabolite: Compound = next(filter(lambda c: c.name == row['Metabolite'], compound_list))
            met_des = MetaboliteDescription(formation_fraction=row['Formation Fraction'], metabolite=metabolite)
            object.__setattr__(parent, 'metabolites', parent.metabolites + (met_des,))
        return parents

    @staticmethod
    def from_path(path: Path) -> Generator['Compound', None, None]:
        if path.is_dir():
            for file in path.iterdir():
                yield from Compound.from_file(file)
        else:
            yield from Compound.from_file(path)

    @staticmethod
    def from_file(file: Path) -> Generator['Compound', None, None]:
        if file.suffix == '.json':
            with file.open() as f:
                json_content = json.load(f)
                try:
                    yield from (Compound(**element) for element in json_content)
                except TypeError:
                    yield Compound(**json_content)
        if file.suffix == '.xlsx':
            yield from Compound.from_excel(file)
