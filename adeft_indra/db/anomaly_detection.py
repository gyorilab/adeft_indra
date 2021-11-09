import gzip
import json
import sqlite3
from contextlib import closing

from adeft_indra.locations import RESULTS_DB_PATH


def gzip_dumps(x):
    return gzip.compress(bytes(x, 'utf-8'))


def gzip_loads(x):
    return gzip.decompress(x).decode('utf-8')


class AnomalyDetectorsManager(object):
    def __init__(self, table_name):
        self.table_name = table_name
        self._setup_table()

    def _setup_table(self):
        make_table_anomaly_detectors = \
            f"""CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    grounding TEXT,
                    agent_text TEXT,
                    num_training_texts INTEGER,
                    anomaly_detector BLOB);
             """
        make_idx_anomaly_detectors_grounding = \
            f"""CREATE INDEX IF NOT EXISTS
                   idx_anomaly_detectors_grounding_agent_text
               ON
                   {self.table_name} (grounding, agent_text);
            """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                for query in [make_table_anomaly_detectors,
                              make_idx_anomaly_detectors_grounding]:
                    cur.execute(query)
            conn.commit()

    def in_table(self, grounding, agent_text):
        query = \
            f"""SELECT
                   grounding, agent_text
               FROM
                   {self.table_name}
               WHERE
                   grounding = ? AND agent_text = ? AND
                   anomaly_detector IS NOT NULL
               LIMIT 1
            """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query, [grounding, agent_text])
                res = cur.fetchone()
        return res

    def save(
            self, grounding, agent_text, num_training_texts, anomaly_detector
    ):
        query = f"""INSERT INTO
                       {self.table_name} (grounding,
                                          agent_text,
                                          num_training_texts,
                                          anomaly_detector)
                   VALUES
                       (?, ?, ?, ?);
                """
        data = gzip_dumps(json.dumps(anomaly_detector.get_model_info()))
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query, [grounding,
                                    agent_text,
                                    num_training_texts,
                                    sqlite3.Binary(data)])
            conn.commit()

    def load(self, grounding, agent_text):
        query = f"""SELECT
                       num_training_texts, anomaly_detector
                   FROM
                       {self.table_name}
                   WHERE
                       grounding = ? AND agent_text = ?
                   LIMIT 1
                """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query, [grounding, agent_text])
                res = cur.fetchone()
        return (res[0], json.loads(gzip_loads(res[1])))


class ResultsManager(object):
    def __init__(self, table_name):
        self.table_name = table_name
        self._setup_table()

    def _setup_table(self):
        make_table_results = \
            f"""CREATE TABLE IF NOT EXISTS {self.table_name} (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   agent_text TEXT,
                   grounding TEXT,
                   num_training_texts INTEGER,
                   num_prediction_texts INTEGER,
                   num_anomalous_texts INTEGER,
                   specificity REAL,
                   std_specificity REAL,
                   UNIQUE(agent_text, grounding)
               );
            """
        make_idx_results_agent_text_grounding = \
            f"""CREATE UNIQUE INDEX IF NOT EXISTS
                   idx_results_agent_text_grounding
               ON
                   {self.table_name} (agent_text, grounding);
            """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                for query in [make_table_results,
                              make_idx_results_agent_text_grounding]:
                    cur.execute(query)
            conn.commit()

    def in_table(self, agent_text, grounding):
        query = f"""SELECT
                       agent_text, grounding
                   FROM
                       {self.table_name}
                   WHERE
                       agent_text = ? AND grounding = ?
                   LIMIT 1
                """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query, [agent_text, grounding])
                res = cur.fetchone()
        return res

    def add_row(self, row):
        insert_row = \
            f"""INSERT OR IGNORE INTO
                {self.table_name} (agent_text, grounding, num_training_texts,
                            num_prediction_texts, num_anomalous_texts,
                            specificity, std_specificity)
               VALUES
                   (?, ?, ?, ?, ?, ?, ?);
            """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(insert_row, row)
            conn.commit()

    def get_results(self):
        query = f"SELECT * FROM {self.table_name}"
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query)
                res = cur.fetchall()
        return res


class GeneralResultsManager(object):
    def __init__(self, table_name):
        self.table_name = table_name
        self._setup_table()

    def _setup_table(self):
        make_table_results = \
            f"""CREATE TABLE IF NOT EXISTS {self.table_name} (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   agent_text TEXT,
                   grounding TEXT,
                   stmt INTEGER,
                   data TEXT,
                   UNIQUE(agent_text, grounding)
               );
            """
        make_idx_results_agent_text_grounding = \
            f"""CREATE UNIQUE INDEX IF NOT EXISTS
                   idx_results_agent_text_grounding
               ON
                   {self.table_name} (agent_text, grounding);
            """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                for query in [make_table_results,
                              make_idx_results_agent_text_grounding]:
                    cur.execute(query)
            conn.commit()

    def in_table(self, agent_text, grounding, stmt):
        query = f"""SELECT
                       agent_text, grounding, stmt
                   FROM
                       {self.table_name}
                   WHERE
                       agent_text = ? AND grounding = ? AND stmt = ?
                   LIMIT 1
                """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query, [agent_text, grounding, stmt])
                res = cur.fetchone()
        return res

    def add_row(self, row):
        insert_row = \
            f"""INSERT OR IGNORE INTO
                {self.table_name} (agent_text, grounding, stmt, data)
               VALUES
                   (?, ?, ?, ?);
            """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(insert_row, row)
            conn.commit()

    def get_results(self):
        query = f"SELECT * FROM {self.table_name}"
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query)
                res = cur.fetchall()
        return res


class ADResultsManager(object):
    def __init__(self, table_name):
        self.table_name = table_name
        self._setup_table()

    def _setup_table(self):
        make_table_results = \
            f"""CREATE TABLE IF NOT EXISTS {self.table_name} (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   model_name TEXT,
                   params TEXT,
                   data TEXT,
                   UNIQUE(model_name, params)
               );
            """
        make_idx_model_name_params = \
            f"""CREATE UNIQUE INDEX IF NOT EXISTS
                   idx_{self.table_name}_model_name_params
               ON
                   {self.table_name} (model_name, params);
            """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                for query in [make_table_results,
                              make_idx_model_name_params]:
                    cur.execute(query)
            conn.commit()

    def in_table(self, model_name, params):
        query = f"""SELECT
                       model_name, params
                   FROM
                       {self.table_name}
                   WHERE
                       model_name = ? AND params = ?
                   LIMIT 1
                """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query, [model_name, params])
                res = cur.fetchone()
        return res

    def add_row(self, row):
        insert_row = \
            f"""INSERT OR IGNORE INTO
                    {self.table_name} (model_name, params, data)
                VALUES
                    (?, ?, ?);
            """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(insert_row, row)
            conn.commit()

    def get_results(self):
        query = f"SELECT * FROM {self.table_name}"
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                for row in cur.execute(query):
                    yield row


class AdeftabilityResultsManager(object):
    def __init__(self, table_name):
        self.table_name = table_name
        self._setup_table()

    def _setup_table(self):
        make_table_results = \
            f"""CREATE TABLE IF NOT EXISTS {self.table_name} (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   agent_text TEXT,
                   score REAL,
                   UNIQUE(agent_text)
               );
            """
        make_idx_results_agent_text_grounding = \
            f"""CREATE UNIQUE INDEX IF NOT EXISTS
                   idx_{self.table_name}_agent_text
               ON
                   {self.table_name} (agent_text);
            """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                for query in [make_table_results,
                              make_idx_results_agent_text_grounding]:
                    cur.execute(query)
            conn.commit()

    def in_table(self, agent_text):
        query = f"""SELECT
                       agent_text
                   FROM
                       {self.table_name}
                   WHERE
                       agent_text = ?
                   LIMIT 1
                """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query, [agent_text])
                res = cur.fetchone()
        return res

    def add_row(self, row):
        insert_row = \
            f"""INSERT OR IGNORE INTO
                {self.table_name} (agent_text, score)
               VALUES
                   (?, ?);
            """
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(insert_row, row)
            conn.commit()

    def get_results(self):
        query = f"SELECT * FROM {self.table_name}"
        with closing(sqlite3.connect(RESULTS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query)
                res = cur.fetchall()
        return res
