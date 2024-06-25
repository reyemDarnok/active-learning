from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Union, Dict
from pathlib import Path
import pandas
import sys
sys.path += [str(Path(__file__).parent.parent)]
from util.datastructures import HashableDict
from util.conversions import map_to_class

@dataclass(frozen=True)
class Degradation:
    '''General Degradation information'''
    system: float
    '''Total System DT50'''
    soil: float
    '''DT50 in soil'''
    surfaceWater: float
    '''DT50 in water'''
    sediment: float
    '''DT50 in sediment'''

    def __post_init__(self):
        object.__setattr__(self, 'system', float(self.system))
        object.__setattr__(self, 'soil', float(self.soil))
        object.__setattr__(self, 'surfaceWater', float(self.surfaceWater))
        object.__setattr__(self, 'sediment', float(self.sediment))


@dataclass(frozen=True)
class Sorption:
    '''Information about the sorption behavior of a compound. Steps12 uses the koc, Pelmo uses all values'''
    koc: float
    freundlich: float

    def __post_init__(self):
        object.__setattr__(self, 'koc', float(self.koc))
        object.__setattr__(self, 'freundlich', float(self.freundlich))

@dataclass(frozen=True)
class Volatility:
    water_solubility: float
    vaporization_pressure: float
    reference_temperature: float


@dataclass(frozen=True)
class MetaboliteDescription:
    formation_fraction: float
    metabolite: 'Compound'

@dataclass(frozen=True)
class Compound:
    '''A Compound definition'''
    molarMass: float
    '''molar mass in g/mol'''
    volatility: Volatility
    sorption: Sorption
    '''A sorption behaviour'''
    degradation: Degradation
    '''Degradation behaviours'''
    plant_uptake: float = 0
    '''Fraction of plant uptake'''
    name: str = "Unknown Name"
    model_specific_data: Dict = field(hash=False, default_factory=HashableDict)
    metabolites: Set[MetaboliteDescription] = field(default_factory=frozenset)
    '''The compounds metabolites'''

    def __post_init__(self):
        object.__setattr__(self, 'molarMass', float(self.molarMass))
        object.__setattr__(self, 'plant_uptake', float(self.plant_uptake))
        object.__setattr__(self, 'volatility', map_to_class(self.volatility, Volatility))
        object.__setattr__(self, 'sorption', map_to_class(self.sorption, Sorption))
        object.__setattr__(self, 'degradation', map_to_class(self.degradation, Degradation))
        if self.metabolites:
            object.__setattr__(self, 'metabolites', frozenset([map_to_class(met_des, MetaboliteDescription) for met_des in self.metabolites]))
        else:
            object.__setattr__(self, 'metabolites', frozenset())



    def excel_to_compounds(excel_file: Path) -> List['Compound']:
        compounds = pandas.read_excel(io=excel_file, sheet_name = "Compound Properties")
        metabolite_relationships = pandas.read_excel(io=excel_file, sheet_name="Metabolite Relationships")
        compound_list = [Compound(  name=row['Name'], molarMass=row['Molar Mass'], 
                                    volatility=Volatility(water_solubility=row['Water Solubility'], 
                                                        vaporization_pressure=row['Vaporization Pressure'], 
                                                        reference_temperature=row['Temperature']),
                                    sorption=Sorption(koc=row['Koc'], freundlich=row['Freundlich']),
                                    plant_uptake=row['Plant Uptake'],
                                    degradation=Degradation(system=row['DT50 System'],
                                                            soil=row['DT50 Soil'],
                                                            sediment=row['DT50 Sediment'],
                                                            surfaceWater=row['DT50 Surface Water']),
                                    model_specific_data={'pelmo': {'position': row['Pelmo Position']}}) for _, row in compounds.iterrows()]
        parents = [compound for compound in compound_list if compound.name not in metabolite_relationships['Metabolite'].values]
        for _, row in metabolite_relationships.iterrows():
            parent: Compound = next(filter(lambda c: c.name == row['Parent'], compound_list))
            metabolite: Compound = next(filter(lambda c: c.name == row['Metabolite'], compound_list))
            met_des = MetaboliteDescription(formation_fraction=row['Formation Fraction'], metabolite=metabolite)
            object.__setattr__(parent, 'metabolites', parent.metabolites.union([met_des]))
        return parents
