import csv
import json
from adeft_indra.db.pmids import get_pmids_for_entity

from indra.literature.pubmed_client import logger as pubmed_client_logger


pmid_map = {}
broken = set()
with open('../../../new_grounding_table.tsv', newline='') as csvfile:
    rows = csv.DictReader(csvfile, delimiter='\t')
    for row in rows:
        grounding = row['grounding']
        if not grounding:
            continue
        ns, id_ = grounding.split(':', maxsplit=1)
        if grounding not in pmid_map:
            try:
                pmids = get_pmids_for_entity(ns, id_, major_topic=True)
            except Exception:
                broken.add(grounding)
            pmid_map[grounding] = pmids

with open('../../results/cord19_entity_pmids4.json', 'w') as f:
    json.dump(pmid_map, f)

with open('../../results/cord19_entity_pmids_broken4.json', 'w') as f:
    json.dump(list(broken), f)
