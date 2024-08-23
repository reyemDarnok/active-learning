""" A file describing a compound and its components
"""
import json
from collections import UserList
from dataclasses import dataclass, field
from math import floor, log10
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Dict

import pandas

from focusStepsPelmo.util.conversions import round_property, round_property_sig
from focusStepsPelmo.util.datastructures import TypeCorrecting


@dataclass(frozen=True)
class DT50(TypeCorrecting):
    """General DT50 information"""
    system: float
    """Total System DT50 in days"""
    soil: float
    """DT50 in soil in days"""
    surfaceWater: float
    """DT50 in water in days"""
    sediment: float
    """DT50 in sediment in days"""

    def __post_init__(self):
        super().__post_init__()
        round_property(self, 'system', 2)
        round_property(self, 'soil', 2)
        round_property(self, 'surfaceWater', 2)
        round_property(self, 'sediment', 2)

@dataclass(frozen=True)
class MetaboliteDescription(TypeCorrecting):
    """A structure describing a decay to a metabolite"""
    formation_fraction: float
    """How much of all dt50 forms this metabolite. Needs to be between 0 and 1. Unitless"""
    metabolite: 'Compound'
    """The Compound that forms"""

    def __post_init__(self):
        super().__post_init__()
        round_property(self, 'formation_fraction', 4)



@dataclass(frozen=True)
class Compound(TypeCorrecting):
    """A Compound definition"""
    molarMass: float
    """molar mass in g/mol"""
    water_solubility: float
    """The water solubility in mg/L"""
    vapor_pressure: float
    """The vaporization pressure in Pa"""
    reference_temperature: float
    """The temperature the other values have been measured at in °C"""
    koc: float
    """The sorption with organic compounds in ml/g"""
    freundlich: float
    """The freundlich exponent. Unitless"""
    dt50: DT50
    """DT50 behaviours"""
    plant_uptake: float = 0
    """Fraction of plant uptake. Needs to between 0 and 1. Unitless"""
    name: Optional[str] = field(hash=False, default='')  # str hash is not stable
    """The compounds name. Used only for labelling purposes"""
    model_specific_data: Dict = field(compare=False, hash=False, default_factory=dict)
    """Some data only of interest to specific models"""
    metabolites: Tuple[MetaboliteDescription, ...] = field(default_factory=tuple)
    """The compounds metabolites"""

    def __post_init__(self):
        super().__post_init__()
        # To ensure a stable, unique name for a compound that has no name given
        if self.name == '':
            object.__setattr__(self, 'name', f"Compound {hash(self)}")
        round_property(self, 'freundlich', 5)
        round_property(self, 'koc', 2)
        round_property(self, 'plant_uptake', 4)
        round_property_sig(self, 'vapor_pressure', 2)
        round_property(self, 'water_solubility', 4)


    def metabolite_description_by_name(self, name: str) -> Optional[MetaboliteDescription]:
        """Given a name, find the MetaboliteDescription for the Metabolite with that name
        :param name: The name of the Metabolite to find
        :return: The MetaboliteDescription for the degradation from self to the Metabolite named name"""
        if self.metabolites is not None:
            for met_des in self.metabolites:
                if met_des.metabolite.name == name:
                    return met_des
        return None

    @staticmethod
    def from_excel(excel_file: Path) -> List['Compound']:
        """Parse an Excel file and find all compounds defined in it
        :param excel_file: The Excel File to parse
        :return: The Compounds in the file"""
        compounds = pandas.read_excel(io=excel_file, sheet_name="Compound Properties")
        compounds['Pelmo Position'].fillna('', inplace=True)
        metabolite_relationships = pandas.read_excel(io=excel_file, sheet_name="Metabolite Relationships")
        compound_list = [
            Compound(name=row['Name'], molarMass=row['Molar Mass'],
                     water_solubility=row['Water Solubility'],
                     vapor_pressure=row['Vapor Pressure'],
                     reference_temperature=row['Temperature'],
                     koc=row['Koc'], freundlich=row['Freundlich'],
                     plant_uptake=row['Plant Uptake'],
                     dt50=DT50(system=row['DT50 System'],
                               soil=row['DT50 Soil'],
                               sediment=row['DT50 Sediment'],
                               surfaceWater=row['DT50 Surface Water']),
                     model_specific_data={'pelmo': {'position': row['Pelmo Position'] if row['Pelmo Position'] else None
                                                    }
                                          }
                     )
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
        """Parse all Compounds found in path
        :param path: The Path to traverse to find possible Compound definitions
        :return: All Compounds that could be found in path"""
        if path.is_dir():
            for file in path.iterdir():
                yield from Compound.from_file(file)
        else:
            yield from Compound.from_file(path)

    @staticmethod
    def from_file(file: Path) -> Generator['Compound', None, None]:
        """Parse a file to compound definitions
        :param file: The file to parse
        :return: All Compounds that are defined in the file"""
        if file.suffix == '.json':
            with file.open() as f:
                json_content = json.load(f)
                if isinstance(json_content, (list, UserList)):
                    yield from (Compound(**element) for element in json_content)
                else:
                    yield Compound(**json_content)
        if file.suffix == '.xlsx':
            yield from Compound.from_excel(file)
