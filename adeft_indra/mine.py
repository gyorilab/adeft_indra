import os
import re
import uuid
import json
import logging
from collections import Counter
from multiprocessing import Pool
from sqlalchemy import text as sqltext


from indra_db.util import get_primary_db
from indra.literature.adeft_tools import universal_extract_text
from indra_db.util.content_scripts import get_text_content_from_trids

from adeft.discover import AdeftMiner, load_adeft_miner, compose


def _get_all_trids():
    db = get_primary_db()
    query = 'SELECT id from text_ref'
    res = db.session.execute(sqltext(query))
    return res.fetchall()


class MiningOperation(object):
    def __init__(self, outpath, trids, batch_size=1000):
        self.outpath = outpath
        self.trids = trids
        self.batch_size = batch_size
        self.current_batch = 0
        self.miners = {}
        pattern = r'(?<=\()\s?\w+(?=\s?\))'
        self.defining_pattern_pattern = re.compile(pattern)
        self.banned_prefixes = ('Figure', 'Table', 'Fig')

    def find_shortforms(self, texts):
        shortforms = Counter()
        for text in texts:
            matches = re.findall(self.defining_pattern_pattern, text)
            for match in matches:
                shortform = match.strip()
                if 2 <= len(shortform) <= 10:
                    shortforms[shortform] += 1
        return shortforms

    def stream_raw_texts(self):
        for i in range(self.current_batch, len(self.trids), self.batch_size):
            trid_batch = self.trids[i:i+self.batch_size]
            _, content_dict = get_text_content_from_trids(trid_batch)
            texts = [universal_extract_text(content) for content in
                     content_dict.values() if content]
            yield texts

    def dump_miners(self, current_batch=0):
        filename_dict = {shortform: value[1]
                         for shortform, value in self.miners.items()}
        with open(os.path.join(self.outpath, 'filenames.json'), 'w') as f:
            json.dump({'filenames': filename_dict,
                       'current_batch': current_batch}, f)
        for _, (miner, filename) in self.miners.items():
            with open(os.path.join(self.outpath, filename), 'w') as f:
                miner.dump(f)

    def load_miners(self):
        with open(os.path.join(self.outpath, 'filenames.json'), 'r') as f:
            filename_dict, current_batch = json.load(f)
        self.current_batch = current_batch
        for shortform, filename in filename_dict.items():
            with open(os.path.join(self.outpath, filename), 'r') as f:
                self.miners[shortform] = load_adeft_miner(f)

    def mine(self):
        texts_stream = self.stream_raw_texts()
        for texts in texts_stream:
            pass
