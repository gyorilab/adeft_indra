import csv
import json
import famplex
from collections import defaultdict
from indra.ontology.bio import bio_ontology
from gilda.resources import MESH_MAPPINGS_PATH
from indra.databases.hgnc_client import get_hgnc_name
from pyobo.sources.pubchem import logger as pyobo_logger
from pyobo.sources.pubchem import get_pubchem_id_to_mesh_id
from indra.databases.chebi_client import get_chebi_id_from_pubchem

from indra.literature.pubmed_client import get_ids_for_gene, get_ids_for_mesh

pyobo_logger.setLevel('ERROR')


class MeshMapper(object):
    def __init__(self):
        # get gilda MESH equivalences (excluding famplex, for which an exhaustive
        # set has been added to the famplex package.
        gilda_equiv = defaultdict(set)
        with open(MESH_MAPPINGS_PATH, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                if row[3] == 'FPLX':
                    continue
                gilda_equiv[f'{row[3]}:{row[4]}'].add(row[1])
        # get famplex equivalences
        fplx_equiv = defaultdict(set)
        fplx_equiv_rows = famplex.load_equivalences()
        for row in fplx_equiv_rows:
            if row[0] == 'MESH':
                fplx_equiv[row[2]].add(row[1])
        # get additional chebi equivalences through pubchem
        chebi_equiv = defaultdict(set)
        pubchem_equiv = get_pubchem_id_to_mesh_id()
        for pubchem_id, mesh_id in pubchem_equiv.items():
            chebi_id = get_chebi_id_from_pubchem(pubchem_id)
            if chebi_id is not None:
                chebi_equiv[chebi_id].add(mesh_id)
        self.gilda_equiv = gilda_equiv
        self.fplx_equiv = fplx_equiv
        self.chebi_equiv = chebi_equiv
        bio = bio_ontology
        bio.initialize()
        self.bio = bio

    def map_to_mesh(self, namespace, id_):
        mesh_ids = set()
        if namespace == 'FPLX':
            equivs = self.fplx_equiv.get(id_)
            if equivs is not None:
                mesh_ids.update(equivs)
        elif namespace == 'CHEBI':
            equivs = self.chebi_equiv.get(id_)
            if equivs is not None:
                mesh_ids.update(equivs)
        equivs = self.gilda_equiv.get(f'{namespace}:{id_}')
        if equivs is not None:
            mesh_ids.update(equivs)
        if not mesh_ids:
            mappings = self.bio.get_mappings(namespace, id_)
            for mapped_ns, mapped_id in mappings:
                if mapped_ns == 'MESH':
                    mesh_ids.add(mapped_id)
        return list(mesh_ids)

    
_mesh_mapper = MeshMapper()


def get_pmids_for_entity(namespace, id_):
    if namespace == 'MESH':
        return get_ids_for_mesh(id_, major_topic=True)
    pmids = set()
    mesh_ids = _mesh_mapper.map_to_mesh(namespace, id_)
    for id_ in mesh_ids:
        pmids.update(get_ids_for_mesh(id_, major_topic=True))
    if namespace == 'HGNC':
        pmids.update(get_ids_for_gene(get_hgnc_name(id_)))
    return list(pmids)
            
