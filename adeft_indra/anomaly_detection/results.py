from contextlib import closing
import logging
import pickle
import sqlite3
from Typing import Any, Iterator, List, Tuple


from adeft_indra.locations import RESULTS_DB_PATH

logger = logging.getLogger(__file__)


class ResultsManager:
    """Singleton managing an sqlite database for storing results.

    Results tables function as key, value stores. Keys should be strings
    and values can be anything that is pickle serializable.

    Within a table, only one entry can be stored for a given key at any
    one time. The main use case is situations where there are many jobs
    to be run parametrized by a set of parameters, i.e. for each combination
    of parameters there is one and only one job to run. Each parameter
    combination is encoded into a key for which the results corresponding
    to the parameter combination are stored in the table. In case of error,
    system crash, or termination of a cloud instance, the batch of jobs can
    be restarted without recomputing those for any parameter combinations that
    are already in the results table.
    """
    @classmethod
    def add_table(cls, table: str) -> None:
        """Add a key, value table to the database."""
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

    @classmethod
    def show_tables(cls) -> List[str]:
        """Get list of tables currently available."""
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

    @classmethod
    def insert(cls, table: str, key: str, value: Any) -> None:
        """Insert an entry into a table.

        There can only be one entry per key at any given time. User
        will be warned if they try to insert an entry for an existing
        key.
        """
        assert table in cls.show_tables()
        assert isinstance(key, str)
        if cls.get(table, key) is not None:
            logger.warning(f"Value already inserted for key {key}")
            return
        query = f"""--
        INSERT INTO
            {table} (key, value)
        VALUES
            (?, ?);
        """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(
                    query,
                    (
                        key,
                        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                    )
                )

    @classmethod
    def get(cls, table: str, key: str) -> Any:
        assert isinstance(key, str)
        query = f"""--
        SELECT
            value
        FROM
            {table}
        WHERE
            key = ?;
        """
        assert table in cls.show_tables()
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                result = cur.execute(query, (key, )).fetchone()
        if not result:
            return None
        return pickle.loads(result[0])

    @classmethod
    def remove(cls, table: str, key: str) -> None:
        """Remove entry in table associated to a given key."""
        assert isinstance(key, str)
        assert table in cls.show_tables()
        query = f"""--
        DELETE FROM
            {table}
        WHERE
            key = ?;
        """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query, (key, ))

    @classmethod
    def iterrows(cls, table: str) -> Iterator[Tuple[str, Any]]:
        """Iterate through key, value pairs in a given table."""
        assert table in cls.show_tables()
        query = f"SELECT key, value FROM {table}"
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                for row in cur.execute(query):
                    key, value = row
                    yield key, pickle.loads(value)

    @classmethod
    def drop_table(cls, table: str) -> None:
        """Drop a table."""
        query = f"DROP TABLE {table}"
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query)
