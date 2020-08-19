import os
import csv
import pickle
import sqlite3
from contextlib import closing

from adeft_indra.locations import CACHE_PATH


class AnomalyDetectorsManager(object):
    def __init__(self, cache_path=CACHE_PATH):
        self.cache_path = cache_path
        self._setup_table()

    def _setup_table(self):
        make_table_anomaly_detectors = \
            """CREATE TABLE IF NOT EXISTS anomaly_detectors (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   grounding TEXT,
                   anomaly_detector BLOB);
            """
        make_idx_anomaly_detectors_grounding = \
            """CREATE INDEX IF NOT EXISTS
                   idx_anomaly_detectors_grounding
               ON
                   anomaly_detectors (grounding);
            """
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                for query in [make_table_anomaly_detectors,
                              make_idx_anomaly_detectors_grounding]:
                    cur.execute(query)
            conn.commit()

    def in_table(self, grounding):
        query = \
            """SELECT
                   grounding
               FROM
                   anomaly_detectors
               WHERE
                   grounding = ?
               LIMIT 1
            """
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query, [grounding])
                res = cur.fetchone()
        return res

    def save(self, grounding, anomaly_detector):
        query = """INSERT INTO
                       anomaly_detectors (grounding, anomaly_detector)
                   VALUES
                       (?, ?);
                """
        pdata = pickle.dumps(anomaly_detector, pickle.HIGHEST_PROTOCOL)
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query, [grounding, sqlite3.Binary(pdata)])
            conn.commit()

    def load(self, grounding):
        query = """SELECT
                       anomaly_detector
                   FROM
                       anomaly_detectors
                   WHERE
                       grounding = ?
                   LIMIT 1
                """
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query, [grounding])
                res = cur.fetchone()[0]
        return pickle.loads(res)


class ResultsManager(object):
    def __init__(self, cache_path=CACHE_PATH):
        self.cache_path = CACHE_PATH
        self._setup_table()

    def _setup_table(self):
        make_table_results = \
            """CREATE TABLE IF NOT EXISTS results (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   agent_text TEXT,
                   grounding TEXT,
                   num_training_texts INTEGER,
                   num_prediction_texts INTEGER,
                   num_anomalous_texts INTEGER,
                   specificity REAL,
                   std_specifity REAL,
                   UNIQUE(agent_text, grounding)
               );
            """
        make_idx_results_agent_text_grounding = \
            """CREATE UNIQUE INDEX IF NOT EXISTS
                   idx_results_agent_text_grounding
               ON
                   results (agent_text, grounding);
            """
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                for query in [make_table_results,
                              make_idx_results_agent_text_grounding]:
                    cur.execute(query)
            conn.commit()

    def in_table(self, agent_text, grounding):
        query = """SELECT
                       agent_text, grounding
                   FROM
                       results
                   WHERE
                       agent_text = ? AND grounding = ?
                   LIMIT 1
                """
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query, [agent_text, grounding])
                res = cur.fetchone()
        return res

    def add_row(self, row):
        insert_row = \
            """INSERT OR IGNORE INTO
                   results (agent_text, grounding, num_training_texts,
                            num_training_texts, num_prediction_texts,
                            num_anomalous_texts, specificity, std_specificity)
               VALUES
                   (?, ?, ?, ?, ?, ?, ?);
            """
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(insert_row, row)
            conn.commit()

    def dump(self, outpath):
        outpath = os.path.realpath(os.path.expanduser(outpath))
        query = "SELECT * FROM results"
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(query)
                with open(outpath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    for row in cur:
                        writer.writewrow(row)
