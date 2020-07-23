import os
import json
import famplex
from gilda import ground
from indra.literature.adeft_tools import universal_extract_text
from indra.databases.hgnc_client import get_hgnc_name, get_hgnc_id
from indra_db.util.content_scripts import get_text_content_from_pmids
from indra.literature.pubmed_client import get_ids_for_gene, get_ids_for_mesh


with open('../../data/synonyms_dict.json') as f:
    synonyms = json.load(f)


def find_groundings(shortform):
    groundings = [(grounding.term.db, grounding.term.id)
                  for grounding in ground(shortform)
                  if grounding.match.exact]
    return groundings


def get_entity_texts_for_grounding(ns, id_):
    return synonyms.get(f'{ns}:{id_}')


def get_text_content_for_entity(ns, id_):
    if ns == 'HGNC':
        symbol = get_hgnc_name(id_)
        pmids = get_ids_for_gene(symbol)
    elif ns == 'MESH':
        pmids = get_ids_for_mesh(id_, major_topic=False)
    elif ns == 'FPLX':
        individual_genes = famplex.individual_members(ns, id_)
        pmids = set(pmid for _, symbol in individual_genes
                    for pmid in get_ids_for_gene(symbol) if pmid)
        gene_ids = [get_hgnc_id(symbol) for _, symbol in individual_genes]
        synonyms = set(synonym for gene_id in gene_ids
                       for synonym in get_entity_texts_for_grounding('HGNC',
                                                                     gene_id))
        synonyms.update(get_entity_texts_for_grounding('FPLX', id_))
        equivalences = famplex.equivalences(id_)
        for equiv_ns, equiv_id in equivalences:
            if equiv_ns == 'MESH':
                pmids.update([pmid for pmid in
                              get_ids_for_mesh(equiv_id, major_topic=False)
                             if pmid])
                synonyms.update(get_entity_texts_for_grounding('MESH',
                                                               equiv_id))
    else:
        return []
    synonyms = get_entity_texts_for_grounding(ns, id_)
    _, xmls = get_text_content_from_pmids(pmids)
    texts = [universal_extract_text(text, contains=synonyms)
             for text in xmls.values() if text]
    texts = [text for text in texts if text != '\n']
    return texts
