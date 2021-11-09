import gzip
import logging
import sqlite3
from contextlib import closing

from adeft_indra.locations import RESULTS_DB_PATH

logger = logging.getLogger(__file__)


def gzip_dumps(x):
    return gzip.compress(bytes(x, 'utf-8'))


def gzip_loads(x):
    return gzip.decompress(x).decode('utf-8')


class KeyValueStoreManager:
    def __init__(self):
        self._setup_table()

    def create_table(self, table):
        table_query = f"""--
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT,
            value BLOB,
            UNIQUE(key)
        );
        """
        index_query = f"""--
        CREATE INDEX IF NOT EXISTS
            {table}_idx
        ON
            {table} (key);
        """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                for query in table_query, index_query:
                    cur.execute(query)

    def get_tables(self):
        query = """--
        SELECT
            name
        FROM
            sqlite_master
        WHERE
            type = 'table' AND
            name NOT LIKE 'sqlite_%'
        """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                result = cur.execute(query).fetchall()
        if not result:
            return []
        return [row[0] for row in result]

    def insert(self, table, key, value):
        if self.get(table, key) is not None:
            logger.info(f"Value already inserted for key {key}")
        query = f"""--
        INSERT INTO
            {table} (key, value)
        VALUES
            (?, ?);
        """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query, (key, gzip_dumps(value)))

    def get(self, table, key):
        query = f"""--
        SELECT
            value
        FROM
            {table}
        WHERE
            key = ?;
        """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                result = cur.execute(query, (key, ))
        if not result:
            return None
        return gzip_loads(result[0][0])
