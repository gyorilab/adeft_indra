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
    def __init__(self, maxsize, location, getsizeof=None):
        super().__init__(maxsize, getsizeof)
        self.location = location
        self.filenames = {}

    def popitem(self):
        key, value = super().popitem()
        with open(os.path.join(self.location, self.filenames[key]), 'w') as f:
            value.dump(f)
        return key, value

    def __missing__(self, key):
        if key not in self.filenames:
            miner = AdeftMiner(key)
            filename = uuid.uuid1().hex
            self.filenames[key] = filename
            with open(os.path.join(self.location, filename), 'w') as f:
                miner.dump(f)
        else:
            with open(os.path.join(self.location,
                                   self.filenames[key]), 'r') as f:
                miner = load_adeft_miner(f)
        self[key] = miner
        return miner

    def save_all(self):
        with open(os.path.join(self.location,
                               'filenames.json'), 'w') as f:
            json.dump(self.filenames, f)
        for key, miner in self.items():
            with open(os.path.join(self.location,
                                   self.filenames[key]), 'w') as f:
                miner.dump(f)

    def reload(self):
        try:
            with open(os.path.join(self.location, 'filenames.json'), 'r') as f:
                self.filenames = json.load(f)
        except FileNotFoundError:
            pass


def _get_all_trids():
    db = get_primary_db()
    query = 'SELECT id from text_ref'
    res = db.session.execute(sqltext(query))
    return res.fetchall()


class MiningOperation(object):
    def __init__(self, outpath, name, trids, batch_size=1000,
                 pattern=r'(?<=\()\s?\w+(?=\s?\))',
                 shortform_size=(2, 10), filter_function=None,
                 cache_size=1000):
        self.name = name
        self.miners = MinerCache(cache_size, os.path.join(outpath, name))
        self.trids = trids
        self.batch_size = batch_size
        self.defining_pattern_pattern = re.compile(pattern)
        self.banned_prefixes = ('Figure', 'Table', 'Fig')
        if filter_function is None:
            def filter_function(shortform):
                return True if \
                    shortform_size[0] <= shortform <= shortform_size[1] \
                    else False
        self.filter_function = filter_function
        logger = logging.getLogger(name)
        file_handler = logging.fileHandler(os.path.join(outpath, 'mine.log'))
        log_format = logging.Formatter('%(asctime)s - %(name)s - '
                                       '%(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        self.logger = logger

    def find_shortforms(self, text):
        shortforms = set([])
        matches = re.findall(self.defining_pattern_pattern, text)
        for match in matches:
            shortform = match.strip()
            if self.filter_function(shortform):
                shortforms.add(shortform)
        return shortforms

    def stream_raw_texts(self):
        for i in range(0, len(self.trids), self.batch_size):
            trid_batch = self.trids[i:i+self.batch_size]
            ref_dict, content_dict = get_text_content_from_trids(trid_batch)
            content = [(trid, content_dict[ref_dict[trid]])
                       for trid in ref_dict]
            texts = [(trid, universal_extract_text(text)) for
                     trid, text in content if text]
            yield (i, texts)

    def mine(self):
        texts_stream = self.stream_raw_texts()
        for batch_number, texts in texts_stream:
            for trid, text in texts:
                try:
                    shortforms = self.find_shortforms(text)
                    for shortform in shortforms:
                        miner = self.miners[shortform]
                        miner.process_texts([text])
                except Exception:
                    self.miners.save_all()
                    self.logger.error(f'Failure in batch {batch_number}'
                                      f' for text_ref_id {trid} with'
                                      f' text:\n{text[0:3000]}...')
            self.miners.save_all()
