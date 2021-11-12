import argparse
from contextlib import closing
from itertools import zip_longest
import sqlite3

from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer

from famplex import load_equivalences

from indra_db_lite import get_plaintexts_for_text_ref_ids
from indra_db_lite import get_pmids_for_mesh_term
from indra_db_lite import get_text_ref_ids_for_pmids
from indra_db_lite.locations import INDRA_DB_LITE_LOCATION


def get_all_entrez_pmids():
    query = "SELECT DISTINCT pmid FROM entrez_pmids"
    with closing(sqlite3.connect(INDRA_DB_LITE_LOCATION)) as conn:
        with closing(conn.cursor()) as cur:
            result = set(row[0] for row in cur.execute(query))
    return result


def get_famplex_mesh_pmids():
    equivalences = load_equivalences()
    famplex_mesh_terms = [id_ for ns, id_, _ in equivalences if ns == 'MESH']
    result = set()
    for mesh_term in famplex_mesh_terms:
        result.update(get_pmids_for_mesh_term(mesh_term, major_topic=False))
    return result


def get_uniprot_mesh_pmids():
    query = """--
    SELECT
        mpmid.pmid_num
    FROM
        mesh_xrefs mxref
    INNER JOIN
        mesh_pmids mpmid
    ON
        mxref.mesh_num = mpmid.mesh_num AND
        mxref.is_concept = mpmid.is_concept AND
        (mxref.curie LIKE 'UP:%' OR mxref.curie LIKE 'HGNC:%')
    """
    with closing(sqlite3.connect(INDRA_DB_LITE_LOCATION)) as conn:
        with closing(conn.cursor()) as cur:
            result = set(row[0] for row in cur.execute(query))
    return result


def get_trids_for_training_set():
    pmids = (
        get_famplex_mesh_pmids() |
        get_all_entrez_pmids() |
        get_uniprot_mesh_pmids()
    )
    pmid2trid = get_text_ref_ids_for_pmids(pmids)
    return list(pmid2trid.values())


tokenize = TfidfVectorizer().build_tokenizer()


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def preprocess(text):
    return [token.lower() for token in tokenize(text)]


class ContentIterator(object):
    def __init__(self, trid_list, chunksize=10000):
        self.trids = trid_list
        self.chunksize = chunksize

    def __iter__(self):
        groups = grouper(self.trids, self.chunksize)
        for trids in groups:
            texts = get_plaintexts_for_text_ref_ids(trids)
            for text in texts:
                tokens = preprocess(text)
                if not {'xml', 'elsevier', 'doi', 'article'} <= set(tokens):
                    yield tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('outpath')
    args = parser.parse_args()
    outpath = args.outpath
    trids = get_trids_for_training_set()
    all_pubmed_content = ContentIterator(trids)
    dictionary = Dictionary(
        (text for text in all_pubmed_content), prune_at=None
    )
    dictionary.save(outpath)
