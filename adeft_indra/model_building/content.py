import famplex
from gilda import ground
from indra.databases.hgnc_client import get_hgnc_name, get_hgnc_id
from indra_db.util.content_scripts import get_agent_texts_for_pmids
from indra.literature.pubmed_client import get_ids_for_gene, get_ids_for_mesh


def find_groundings(shortform):
    groundings = [(grounding.term.db, grounding.term.id)
                  for grounding in ground(shortform)
                  if grounding.match.exact]
    return groundings


def get_pmids_for_entity(ns, id_):
    if ns == 'HGNC':
        symbol = get_hgnc_name(id_)
        pmids = get_ids_for_gene(symbol)
    elif ns == 'MESH':
        pmids = get_ids_for_mesh(id_, major_topic=True)
    elif ns == 'FPLX':
        pmids = set()
        equivalences = famplex.equivalences(id_)
        for equiv_ns, equiv_id in equivalences:
            if equiv_ns == 'MESH':
                pmids.update([pmid for pmid in
                              get_ids_for_mesh(equiv_id, major_topic=True)
                              if pmid])
        pmids = list(pmids)
    else:
        return []
    return pmids


def get_agent_texts_for_entity(ns, id_):
    pmids = get_pmids_for_entity(ns, id_)
    if not pmids:
        return []
    counts = get_agent_texts_for_pmids(pmids)
    return sorted(counts.items(), key=lambda x: -x[1])
