import zlib
import sqlite3
from contextlib import closing
from indra.literature.adeft_tools import universal_extract_text


from adeft_indra.locations import CONTENT_DB_PATH


def get_plaintexts_for_pmids(pmids, contains=None):
    if contains is None:
        contains = []
    # Find which pmids have associated plaintexts already cached
    xmls = _get_xmls_for_pmids(pmids)
    plaintexts = {pmid: universal_extract_text(xml, contains=contains)
                  for pmid, xml in xmls}
    return plaintexts


def get_pmids_for_agent_text(agent_text):
    query = \
        """SELECT
                pmid
            FROM
                agent_text_pmids
            WHERE agent_text = ?;
        """
    with closing(sqlite3.connect(CONTENT_DB_PATH)) as conn:
        with closing(conn.cursor()) as cur:
            res = cur.execute(query, [agent_text]).fetchall()
    return [row[0] for row in res]


def get_pmids_for_entity(ns, id_, major_topic=False):
    table = 'entity_pmids_major' if major_topic else 'entity_pmids'
    query = \
        f"""SELECT
                pmid
            FROM
                {table}
            WHERE
                grounding = ?;
        """
    with closing(sqlite3.connect(CONTENT_DB_PATH)) as conn:
        with closing(conn.cursor()) as cur:
            res = cur.execute(query,
                              [f'{ns}:{id_}']).fetchall()
    return [row[0] for row in res]


def _get_xmls_for_pmids(pmids):
    pmids = tuple(pmids)
    query = \
        f"""SELECT
                pmid, content
            FROM
                best_content
            WHERE
                pmid IN ({','.join(['?']*len(pmids))})
    """
    with closing(sqlite3.connect(CONTENT_DB_PATH)) as conn:
        with closing(conn.cursor()) as cur:
            res = cur.execute(query, pmids).fetchall()
    return [[pmid, _unpack(bytearray.fromhex(content[2:]))]
            for pmid, content in res]


def _unpack(bts, decode=True):
    ret = zlib.decompress(bts, zlib.MAX_WBITS+16)
    if decode:
        ret = ret.decode('utf-8')
    return ret
