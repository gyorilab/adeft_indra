import os
import csv
import sqlite3
from contextlib import closing

from adeft_indra.locations import CACHE_PATH


class AnomalyDetectionResultsManager(object):
    def __init__(self, cache_path=CACHE_PATH):
        self.cache_path = CACHE_PATH
        self._setup_table()

    def _setup_table(self):
        make_result_table = \
                """CREATE TABLE IF NOT EXISTS results (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       agent_text TEXT,
                       grounding TEXT,
                       num_training_texts INTEGER,
                       num_prediction_texts INTEGER,
                       num_anomalous_texts INTEGER,
                       specificity REAL,
                       std_specifity REAL
                       UNIQUE(agent_text, grounding)
                   );
                """
        with closing(sqlite3.connect(self.cache_path)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(make_result_table)
            conn.commit()

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
