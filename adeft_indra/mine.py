import os
import re
import uuid
import json
import logging
from cachetools import LFUCache
from collections import Counter
from multiprocessing import Pool
from sqlalchemy import text as sqltext


from indra_db.util import get_primary_db
from indra.literature.adeft_tools import universal_extract_text
from indra_db.util.content_scripts import get_text_content_from_trids

from adeft.discover import AdeftMiner, load_adeft_miner, compose


class MinerCache(LFUCache):
    def __init__(self, maxsize, outpath, getsizeof=None):
        super().__init__(maxsize, getsizeof)
        self.outpath = outpath
        self.filenames = {}

    def popitem(self):
        key, value = super().popitem()
        with open(os.path.join(self.outpath, self.filenames[key]), 'w') as f:
            value.dump(f)
        return key, value

    def __missing__(self, key):
        if key not in self.filenames:
            miner = AdeftMiner(key)
            filename = uuid.uuid1().hex
            self.filenames[key] = filename
            with open(os.path.join(self.outpath, filename), 'w') as f:
                miner.dump(f)
        else:
            path = os.path.join(self.outpath, self.filenames[key])
            with open(os.path.join(self.outpath,
                                   self.filenames[key]), 'r') as f:
                miner = load_adeft_miner(f)
        self[key] = miner
        return miner

    def dump_miners(self):
        with open(os.path.join(self.outpath, 'filenames.json'), 'w') as f:
            json.dump(self.filenames, f)
        while self:
            self.popitem()

    def reload(self):
        try:
            with open(os.path.join(self.outpath, 'filenames.json'), 'r') as f:
                self.filenames = json.load(f)
        except FileNotFoundError:
            pass


def _get_all_trids():
    db = get_primary_db()
    query = 'SELECT id from text_ref'
    res = db.session.execute(sqltext(query))
    return res.fetchall()


class MiningOperation(object):
    def __init__(self, outpath, trids, batch_size=1000,
                 pattern=r'(?<=\()\s?\w+(?=\s?\))',
                 shortform_size=(2, 10), filter_function=None,
                 cache_size=1000):
        self.miners = MinerCache(cache_size, outpath)
        self.trids = trids
        self.batch_size = batch_size
        self.current_batch = 0
        self.defining_pattern_pattern = re.compile(pattern)
        self.banned_prefixes = ('Figure', 'Table', 'Fig')
        if filter_function is None:
            def filter_function(shortform):
                return True if \
                    shortform_size[0] <= shortform <= shortform_size[1] \
                    else False
        self.filter_function = filter_function

    def find_shortforms(self, text):
        shortforms = set([])
        matches = re.findall(self.defining_pattern_pattern, text)
        for match in matches:
            shortform = match.strip()
            if self.filter_function(shortform):
                shortforms.add(shortform)
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
