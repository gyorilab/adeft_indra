import json
from famplex import equivalences
from famplex import load_entities

from adeft_indra.model_building.content import get_pmids_for_entity

entities = load_entities()

fplx_mesh_pmids = set()
for entity in entities:
    print(entity)
    try:
        fplx_mesh_pmids |= set(get_pmids_for_entity('FPLX', entity))
    except ValueError:
        print('*******************')
        print(entity)
        print('*******************')

fplx_mesh_pmids = list(fplx_mesh_pmids)
with open('../data/fplx_mesh_pmids.json', 'w') as f:
    json.dump(fplx_mesh_pmids, f)
