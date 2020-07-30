import json

with open('../data/fplx_mesh_pmids.json') as f:
    fplx_mesh_pmids = set(json.load(f))


with open('../data/entrez_all_pmids.json') as f:
    hgnc_entrez_pmids = json.load(f)

hgnc_entrez_pmids = {pmid for pmids in hgnc_entrez_pmids.values()
                     for pmid in pmids}

all_pmids = fplx_mesh_pmids | hgnc_entrez_pmids
