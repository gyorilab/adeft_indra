import sqlite3
from contextlib import closing
from multiprocessing import Pool
from indra.literature.adeft_tools import universal_extract_text
from indra_db.util.content_scripts import get_text_content_from_pmids, \
    get_text_content_from_stmt_ids

from adeft_indra.locations import CACHE_PATH


class ContentCache(object):
    def __init__(self, cache_path=CACHE_PATH):
        self._setup_tables()

    def _load_cached_stmt_ids(self):
        with closing(sqlite3.connect(CACHE_PATH)) as conn:
            cur = conn.cursor()
            select_stmt_ids = \
                """SELECT
                       stmt_id
                   FROM
                       stmts
                """
            with closing(conn.cursor()) as cur:
                cur.execute(select_stmt_ids)
                stmt_ids = cur.fetchall()
        return {row[0] for row in stmt_ids}

    def _load_cached_pmids(self):
        with closing(sqlite3.connect(CACHE_PATH)) as conn:
            select_pmids = \
                """SELECT
                       pmid
                   FROM
                       pmids
                """
            with closing(conn.cursor()) as cur:
                cur.execute(select_pmids)
                pmids = cur.fetchall()
        return {str(row[0]) for row in pmids}

    def _get_text_content_from_stmt_ids_local(self, stmt_ids):
        with closing(sqlite3.connect(CACHE_PATH)) as conn:
            query = \
                f"""SELECT DISTINCT
                        plaintext
                    FROM
                        stmts
                    JOIN
                        content
                    ON
                        stmts.text_ref_id = content.text_ref_id
                    WHERE
                        stmt_id IN ({','.join(['?']*len(stmt_ids))})
                """
            with closing(conn.cursor()) as cur:
                output = list(cur.execute(query, stmt_ids))
        return output

    def _get_text_content_from_pmids_local(self, pmids):
        with closing(sqlite3.connect(CACHE_PATH)) as conn:
            query = \
                f"""SELECT DISTINCT
                        plaintext
                    FROM
                        pmids
                    JOIN
                        content
                    ON
                        pmids.text_ref_id = content.text_ref_id
                    WHERE
                        pmid IN ({','.join(['?']*len(pmids))})
                """
            with closing(conn.cursor()) as cur:
                output = list(cur.execute(query, pmids))
        return output

    def _insert_content(self, content_rows):
        with closing(sqlite3.connect(CACHE_PATH)) as conn:
            content_insert_query = """INSERT OR IGNORE INTO
                                          content (text_ref_id, plaintext)
                                      VALUES
                                          (?, ?);
                                   """
            with closing(conn.cursor()) as cur:
                cur.executemany(content_insert_query, content_rows)
            conn.commit()

    def _insert_stmts(self, stmt_rows):
        with closing(sqlite3.connect(CACHE_PATH)) as conn:
            stmts_insert_query = """INSERT INTO
                                        stmts (stmt_id, text_ref_id)
                                    VALUES
                                        (?, ?);
                                 """
            with closing(conn.cursor()) as cur:
                cur.executemany(stmts_insert_query, stmt_rows)
            conn.commit()

    def _insert_pmids(self, pmid_rows):
        with closing(sqlite3.connect(CACHE_PATH)) as conn:
            pmids_insert_query = """INSERT INTO
                                        pmids (pmid, text_ref_id)
                                    VALUES
                                        (?, ?);
                                 """
            with closing(conn.cursor()) as cur:
                cur.executemany(pmids_insert_query, pmid_rows)
            conn.commit()

    def get_text_content_from_stmt_ids(self, stmt_ids, njobs=1):
        output = []
        precomputed = set(stmt_ids) & self._load_cached_stmt_ids()
        uncached = list(set(stmt_ids) - precomputed)
        if precomputed:
            output = \
                self._get_text_content_from_stmt_ids_local(list(precomputed))
        if uncached:
            ref_dict, content_dict = get_text_content_from_stmt_ids(uncached)
            content_dict = {ref: xml for ref, xml in content_dict.items()
                            if xml}
            plaintexts = extract_plaintext(content_dict.values(), njobs=njobs)
            stmts_rows = [(stmt_id, ref[0])
                          for stmt_id, ref in ref_dict.items()]
            content_rows = [(ref[0], text)
                            for ref, text in
                            zip(content_dict.keys(), plaintexts)]
            output.extend(plaintexts)
            self._insert_stmts(stmts_rows)
            self._insert_content(content_rows)
        return output

    def get_text_content_from_pmids(self, pmids, njobs=1):
        output = []
        precomputed = set(pmids) & self._load_cached_pmids()
        uncached = list(set(pmids) - precomputed)
        if precomputed:
            output = \
                self._get_text_content_from_pmids_local(list(precomputed))
        if uncached:
            ref_dict, content_dict = get_text_content_from_pmids(uncached)
            content_dict = {ref: xml for ref, xml in content_dict.items()
                            if xml}
            plaintexts = extract_plaintext(content_dict.values(), njobs=njobs)
            pmids_rows = [(pmid, ref[0])
                          for pmid, ref in ref_dict.items()]
            content_rows = [(ref[0], text)
                            for ref, text in
                            zip(content_dict.keys(), plaintexts)]
            output.extend(plaintexts)
            self._insert_pmids(pmids_rows)
            self._insert_content(content_rows)
        return output

    def _setup_tables(self):
        with closing(sqlite3.connect(CACHE_PATH)) as conn:
            make_table_content = \
                """CREATE TABLE IF NOT EXISTS content (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       text_ref_id INTEGER NOT NULL,
                       plaintext TEXT,
                       UNIQUE(text_ref_id, plaintext)
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
            make_idx_stmts_stmt_id = \
                """CREATE UNIQUE INDEX IF NOT EXISTS
                       idx_stmts_stmt_id
                   ON
                       stmts (stmt_id);
                """
            make_idx_stmts_text_ref_id = \
                """CREATE INDEX IF NOT EXISTS
                       idx_stmts_text_ref_id
                   ON
                       stmts (text_ref_id);
                """
            make_idx_pmids_pmid = \
                """CREATE UNIQUE INDEX IF NOT EXISTS
                       idx_pmids_pmid
                   ON
                       pmids (pmid);
                """
            make_idx_pmids_text_ref_id = \
                """CREATE UNIQUE INDEX IF NOT EXISTS
                       idx_pmids_pmid_text_ref_id
                   ON
                       pmids (text_ref_id);
                """
            with closing(conn.cursor()) as cur:
                for query in [make_table_content, make_table_stmts,
                              make_table_pmids,
                              make_idx_content_text_ref_id,
                              make_idx_stmts_stmt_id,
                              make_idx_stmts_text_ref_id,
                              make_idx_pmids_pmid,
                              make_idx_pmids_text_ref_id]:
                    cur.execute(query)
            conn.commit()


def extract_plaintext(texts, njobs=1):
    with Pool(njobs) as pool:
        res = pool.map(universal_extract_text, texts)
    return res
