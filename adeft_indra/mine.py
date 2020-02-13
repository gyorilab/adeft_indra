import os
import re
import json
import math
import uuid
import logging
from cachetools import LRUCache
from sqlalchemy import text as sqltext
from pathos.multiprocessing import ProcessingPool as Pool

from indra_db.util import get_primary_db
from indra.literature.adeft_tools import universal_extract_text
from indra_db.util.content_scripts import get_text_content_from_trids

from adeft.discover import AdeftMiner, load_adeft_miner, compose


class MinerCache(LRUCache):
    def __init__(self, maxsize, location, getsizeof=None):
        super().__init__(maxsize, getsizeof)
        self.location = location
        try:
            os.makedirs(location)
        except FileExistsError:
            pass
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

    def dump_all(self):
        with open(os.path.join(self.location,
                               'filenames.json'), 'w') as f:
            json.dump(self.filenames, f)
        while self:
            self.popitem()

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
    return [x[0] for x in res.fetchall()]


class MiningOperation(object):
    def __init__(self, outpath, name, trids, batch_size=10000,
                 pattern=r'(?<=\()\s?\w+(?=\s?\))',
                 shortform_size=(2, 10), filter_function=None,
                 cache_size=1000):
        self.outpath = outpath
        self.name = name
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.trids = trids
        self.defining_pattern_pattern = re.compile(pattern)
        try:
            os.makedirs(outpath)
        except FileExistsError:
            pass
        if filter_function is None:
            def filter_function(shortform):
                return True if \
                    shortform_size[0] <= len(shortform) <= shortform_size[1] \
                    else False
        self.filter_function = filter_function
        logger = logging.getLogger(name)
        file_handler = logging.FileHandler(os.path.join(outpath, 'mine.log'))
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
            yield texts

    def mine(self):
        texts_stream = self.stream_raw_texts()
        for batch_number, texts in enumerate(texts_stream):
            miners = MinerCache(self.cache_size,
                                os.path.join(self.outpath,
                                             f'{self.name}_site-'
                                             '{batch_number}'))
            self.logger.info(f"Batch: {batch_number}")
            for trid, text in texts:
                try:
                    shortforms = self.find_shortforms(text)
                    for shortform in shortforms:
                        miner = miners[shortform]
                        miner.process_texts([text])
                except Exception as e:
                    miners.dump_all()
                    self.logger.error(f'Failure in batch {batch_number}'
                                      f' for text_ref_id {trid} with'
                                      f' text:\n{text[0:3000]}...\n'
                                      f'{e}')
                    raise e
            miners.dump_all()


def strip_mine_the_world(trids, outpath, n_jobs=1, batch_size=5000,
                         cache_size=2000):
    def start_operation(region):
        trids, region_number = region
        mining_op = MiningOperation(outpath, f'operation_{region_number}',
                                    trids,
                                    batch_size=batch_size,
                                    cache_size=cache_size)
        mining_op.mine()
    operation_size = math.ceil(len(trids)/n_jobs)
    regions = [(trids[j:j+batch_size], i) for i, j
               in enumerate(range(0, len(trids), operation_size))]
    with Pool(n_jobs) as pool:
        pool.map(start_operation, regions)


if __name__ == '__main__':
    n_jobs = 1
    all_of_the_content = _get_all_trids()[:10000]
    strip_mine_the_world(all_of_the_content, '/Users/albertsteppi/msc/mine/',
                         n_jobs=n_jobs)
