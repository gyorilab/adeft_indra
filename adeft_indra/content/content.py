import os
import sqlite3
from multiprocessing import Pool
from indra.literature.adeft_tools import universal_extract_text
from indra_db.util.content_scripts import get_stmts_with_agent_text_like, \
    get_text_content_from_stmt_ids, get_text_content_from_pmids

from adeft_indra.locations import CACHE_PATH


class ContentCache(object):
    def __init__(self, cache_path=CACHE_PATH):
        self._setup_tables()

    def _load_cached_stmt_ids(self):
        with sqlite3.connect(CACHE_PATH) as conn:
            cur = conn.cursor()
            select_stmt_ids = \
                """SELECT
                       stmt_id
                   FROM
                       stmts
                """
            cur.execute(select_stmt_ids)
            stmt_ids = set(cur.fetchall())
        return stmt_ids

    def _load_cached_pmids(self):
        with sqlite3.connect(CACHE_PATH) as conn:
            select_pmids = \
                """SELECT
                       pmid
                   FROM
                       pmids
                """
            cur.execute(select_pmids)
            pmids = set(cur.fetchall())
        return pmids

    def get_text_content_from_stmt_ids(self, stmt_ids, njobs=1):
        output = []
        precomputed = set(stmt_ids) & self._load_cached_stmt_ids()
        uncached = list(set(stmt_ids) - precomputed)
        with sqlite3.connect(CACHE_PATH) as conn:
            cur = conn.cursor()
            if precomputed:
                query = \
                    f"""SELECT DISTINCT
                            plaintext
                        FROM
                            stmts
                        JOIN
                            content
                        ON
                            stmts.content_id = content.id
                        WHERE
                            stmt_id IN ({','.join(['?']*len(precomputed))})
                    """
                output = list(cur.execute(query, list(precomputed)))
            if uncached:
                ref_dict, content_dict = \
                    get_text_content_from_stmt_ids(uncached)
                content_dict = {ref: xml for ref, xml in content_dict.items()
                                if xml}
                plaintexts = extract_plaintext(content_dict.values(), njobs=njobs)
                stmts_rows = [(stmt_id, ref[0])
                              for stmt_id, ref in ref_dict.items()]
                content_rows = [(ref[0], text)
                                for ref, text in
                                zip(content_dict.keys(), plaintexts)]
                output.extend(plaintexts)
                stmts_insert_query = """INSERT INTO
                                            stmts (stmt_id, text_ref_id)
                                        VALUES
                                            (?, ?);
                                     """
                cur.executemany(stmts_insert_query, stmts_rows)
                content_insert_query = """INSERT INTO
                                               content (text_ref_id, plaintext)
                                           VALUES
                                               (?, ?);
                                        """
                cur.executemany(content_insert_query, content_rows)
                conn.commit()
        return output
            
    def _setup_tables(self):
        with sqlite3.connect(CACHE_PATH) as conn:
            cur = conn.cursor()
            make_table_content = \
                """CREATE TABLE IF NOT EXISTS content (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       text_ref_id INTEGER NOT NULL,
                       plaintext TEXT
                   );
                """
            make_table_stmts = \
                """CREATE TABLE IF NOT EXISTS stmts (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       stmt_id INTEGER NOT NULL,
                       text_ref_id INTEGER NOT NULL,
                       FOREIGN KEY
                           (text_ref_id) REFERENCES content (text_ref_id)
                   );
                """
            make_table_pmids = \
                """CREATE TABLE IF NOT EXISTS pmids (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       pmid INTEGER NOT NULL,
                       text_ref_id INTEGER NOT NULL,
                       FOREIGN KEY
                           (text_ref_id) REFERENCES content (text_ref_id)
                   );
                """
            make_idx_content_text_ref_id = \
                """CREATE UNIQUE INDEX IF NOT EXISTS
                       idx_content_text_ref_id
                   ON
                       content (text_ref_id);
                """
            make_idx_stmts_stmt_id_text_ref_id = \
                """CREATE UNIQUE INDEX IF NOT EXISTS
                       idx_stmts_stmt_id_text_ref_id
                   ON
                       stmts (stmt_id, text_ref_id);
                """
            make_idx_pmids_pmid_text_ref_id = \
                """CREATE UNIQUE INDEX IF NOT EXISTS
                       idx_pmids_pmid_text_ref_id
                   ON
                       pmids (pmid, text_ref_id);
                """
            for query in [make_table_content, make_table_stmts,
                          make_table_pmids,
                          make_idx_content_text_ref_id,
                          make_idx_stmts_stmt_id_text_ref_id,
                          make_idx_pmids_pmid_text_ref_id]:
                cur.execute(query)
            conn.commit()
        

def extract_plaintext(texts, njobs=1):
    with Pool(njobs) as pool:
        res = pool.map(universal_extract_text, texts)
    return res






