import zlib
import sqlite3
from itertools import chain
from contextlib import closing
from indra.literature.adeft_tools import universal_extract_text


from adeft_indra.locations import CONTENT_DB_PATH
from adeft_indra.locations import PLAINTEXT_CACHE_PATH


def get_plaintexts_for_pmids(pmids):
    # Find which pmids have associated plaintexts already cached
    cached_pmids = _load_cached_pmids(pmids)
    # Load the cached plaintexts
    cached_plaintexts = _load_cached_plaintexts(list(cached_pmids))
    # Load xmls for pmids with plaintexts that have not yet been cached
    # and extract the plaintexts
    uncached_pmids = set(pmids) - cached_pmids
    uncached_xmls = _get_xmls_for_pmids(list(uncached_pmids))
    uncached_plaintexts = [[pmid, universal_extract_text(xml)]
                           for pmid, xml in uncached_xmls]
    # Insert these new plaintexts into the cache
    _insert_content(uncached_plaintexts)
    return {pmid: plaintext for pmid, plaintext in chain(cached_plaintexts,
                                                         uncached_plaintexts)}


def get_pmids_for_agent_text(agent_text):
    query = \
        f"""SELECT
                pmid
            FROM
                agent_text_pmids
            WHERE agent_text = ?;
        """
    with closing(sqlite3.connect(CONTENT_DB_PATH)) as conn:
        with closing(conn.cursor()) as cur:
            res = cur.execute(query, [agent_text]).fetchall()
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



def _load_cached_plaintexts(pmids):
    query = \
        f"""SELECT
                pmid, plaintext
            FROM
                plaintexts
            WHERE
                pmid IN ({','.join(['?']*len(pmids))});
        """
    with closing(sqlite3.connect(PLAINTEXT_CACHE_PATH)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute(query, pmids)
            res = cur.fetchall()
    return list(res)


def _insert_content(content_rows):
    content_insert_query = \
        """INSERT OR IGNORE INTO
               plaintexts (pmid, plaintext)
           VALUES
               (?, ?);
    """
    with closing(sqlite3.connect(PLAINTEXT_CACHE_PATH)) as conn:
        with closing(conn.cursor()) as cur:
            cur.executemany(content_insert_query, content_rows)
        conn.commit()

    
def _load_cached_pmids(pmids):
    select_pmids = \
        f"""SELECT
                pmid
            FROM
                plaintexts
            WHERE
                pmid IN ({','.join(['?']*len(pmids))});
        """
    with closing(sqlite3.connect(PLAINTEXT_CACHE_PATH)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute(select_pmids, pmids)
            res = cur.fetchall()
    return {row[0] for row in res}


def _unpack(bts, decode=True):
    ret = zlib.decompress(bts, zlib.MAX_WBITS+16)
    if decode:
        ret = ret.decode('utf-8')
    return ret

