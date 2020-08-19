import sqlite3
from contextlib import closing
from multiprocessing import Pool
from indra.literature.adeft_tools import universal_extract_text
from indra_db.util.content_scripts import get_content_identifiers_from_pmids, \
    get_content_identifiers_from_stmt_ids, _get_text_content


from adeft_indra.locations import CACHE_PATH


class ContentCache(object):
    def __init__(self, cache_path=CACHE_PATH):
        self.cache_path = CACHE_PATH
        self._setup_tables()

    def _load_cached_stmt_ids(self, stmt_ids):
        select_stmt_ids = \
            f"""SELECT
                    stmt_id
                FROM
                    stmts
                WHERE
                    stmt_id IN ({','.join(['?']*len(stmt_ids))}); 
            """
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(select_stmt_ids, stmt_ids)
                stmt_ids = cur.fetchall()
        return {row[0] for row in stmt_ids}

    def _load_cached_pmids(self, pmids):
        select_pmids = \
            f"""SELECT
                    pmid
                FROM
                    pmids
                WHERE
                    pmid IN ({','.join(['?']*len(pmids))});
            """
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(select_pmids,
                            [int(pmid) for pmid in pmids])
                res = cur.fetchall()
        return {str(row[0]) for row in res}

    def _load_cached_text_ref_ids(self, text_ref_ids):
        select_text_ref_ids = \
            f"""SELECT
                    text_ref_id
                FROM
                    content
                WHERE
                    text_ref_id IN ({','.join(['?']*len(text_ref_ids))});
            """
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(select_text_ref_ids, text_ref_ids)
                text_ref_ids = cur.fetchall()
        return {(row[0]) for row in text_ref_ids}

    def _get_text_content_from_stmt_ids_local(self, stmt_ids):
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
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                res = cur.execute(query, stmt_ids).fetchall()
        return [row[0] for row in res]

    def _get_text_content_from_pmids_local(self, pmids):
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
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                res = cur.execute(query, pmids).fetchall()
        return [row[0] for row in res]

    def _get_text_content_from_text_ref_ids_local(self, text_ref_ids):
        query = \
            f"""SELECT DISTINCT
                    plaintext
                FROM
                    content
                WHERE
                    text_ref_id IN ({','.join(['?']*len(text_ref_ids))})
            """
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                res = cur.execute(query, text_ref_ids).fetchall()
        return [row[0] for row in res]

    def _insert_content(self, content_rows):
        content_insert_query = """INSERT OR IGNORE INTO
                                      content (text_ref_id, plaintext)
                                  VALUES
                                      (?, ?);
                               """
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                cur.executemany(content_insert_query, content_rows)
            conn.commit()

    def _insert_stmts(self, stmt_rows):
        stmts_insert_query = """INSERT INTO
                                    stmts (stmt_id, text_ref_id)
                                VALUES
                                    (?, ?);
                             """
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                cur.executemany(stmts_insert_query, stmt_rows)
            conn.commit()

    def _insert_pmids(self, pmid_rows):
        pmids_insert_query = """INSERT INTO
                                    pmids (pmid, text_ref_id)
                                VALUES
                                    (?, ?);
                             """
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                cur.executemany(pmids_insert_query, pmid_rows)
            conn.commit()

    def get_text_content_from_stmt_ids(self, stmt_ids, njobs=1):
        output = []
        cached_stmt_ids = self._load_cached_stmt_ids(stmt_ids)
        uncached_stmt_ids = list(set(stmt_ids) - cached_stmt_ids)
        cached_stmt_ids = list(cached_stmt_ids)
        if cached_stmt_ids:
            output = \
                self._get_text_content_from_stmt_ids_local(cached_stmt_ids)
        if uncached_stmt_ids:
            idf_dict = get_content_identifiers_from_stmt_ids(uncached_stmt_ids)
            text_ref_ids = [idf[0] for idf in idf_dict.values()]
            cached_trids = self._load_cached_text_ref_ids(text_ref_ids)
            uncached_trids = list(set(text_ref_ids) - cached_trids)
            cached_trids = list(cached_trids)
            stmt_rows = []
            if cached_trids:
                output.\
                    extend(self.
                           _get_text_content_from_text_ref_ids_local(
                               cached_trids))
                stmt_rows = [(stmt_id, ref[0])
                             for stmt_id, ref in idf_dict.items()
                             if ref[0] in cached_trids]
            if uncached_trids:
                identifiers = [idf for idf in idf_dict.values()
                               if idf[0] in uncached_trids]
                content_dict = _get_text_content(identifiers)
                plaintexts = extract_plaintext(content_dict.values(),
                                               njobs=njobs)
                content_rows = [(ref[0], text)
                                for ref, text in
                                zip(content_dict.keys(), plaintexts)]
                stmt_rows = [(stmt_id, ref[0])
                             for stmt_id, ref in idf_dict.items()
                             if ref[0] in uncached_trids]
                output.extend(plaintexts)
                self._insert_content(content_rows)
            if stmt_rows:
                self._insert_stmts(stmt_rows)
        return output

    def get_text_content_from_pmids(self, pmids, njobs=1):
        output = []
        cached_pmids = self._load_cached_pmids(pmids)
        uncached_pmids = list(set(pmids) - cached_pmids)
        cached_pmids = list(cached_pmids)
        if cached_pmids:
            output = \
                self._get_text_content_from_pmids_local(cached_pmids)
        if uncached_pmids:
            idf_dict = get_content_identifiers_from_pmids(uncached_pmids)
            text_ref_ids = [idf[0] for idf in idf_dict.values()]
            cached_trids = self._load_cached_text_ref_ids(text_ref_ids)
            uncached_trids = list(set(text_ref_ids) - cached_trids)
            cached_trids = list(cached_trids)
            pmid_rows = []
            if cached_trids:
                output.\
                    extend(self.
                           _get_text_content_from_text_ref_ids_local(
                               cached_trids))
                pmid_rows = [(pmid, ref[0])
                             for pmid, ref in idf_dict.items()
                             if ref[0] in cached_trids]
            if uncached_trids:
                identifiers = [idf for idf in idf_dict.values()
                               if idf[0] in uncached_trids]
                content_dict = _get_text_content(identifiers)
                plaintexts = extract_plaintext(content_dict.values(),
                                               njobs=njobs)
                content_rows = [(ref[0], text)
                                for ref, text in
                                zip(content_dict.keys(), plaintexts)]
                pmid_rows = [(pmid, ref[0])
                             for pmid, ref in idf_dict.items()
                             if ref[0] in uncached_trids]
                output.extend(plaintexts)
                self._insert_content(content_rows)
            if pmid_rows:
                self._insert_stmts(pmid_rows)
        return output

    def _setup_tables(self):
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
        with closing(sqlite3.connect(self.cache_path)) as conn:
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
