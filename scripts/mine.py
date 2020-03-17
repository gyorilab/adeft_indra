import os
import re
import json
import uuid
import random
import logging
from cachetools import LRUCache
from multiprocessing.pool import Pool


from indra.literature.adeft_tools import universal_extract_text

from indra_db.util.helpers import unpack

from adeft.discover import AdeftMiner, load_adeft_miner
from adeft_indra.content import get_texts


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


class MiningOperation(object):
    def __init__(self, outpath, name,
                 pattern=r'(?<=\()\s?\w+(?=\s?\))',
                 shortform_size=(2, 10), filter_function=None,
                 cache_size=1000):
        self.outpath = outpath
        self.name = name
        self.cache_size = cache_size
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

    def __call__(self, content_list):
        miners = MinerCache(self.cache_size,
                            os.path.join(self.outpath,
                                         f'{self.name}', 'miners'))
        processed = {}
        for tcid, content in content_list:
            try:
                text = universal_extract_text(unpack(content))
                shortforms = self.find_shortforms(text)
                for shortform in shortforms:
                    miner = miners[shortform]
                    miner.process_texts([text])
                if shortforms:
                    processed[tcid] = list(shortforms)
            except Exception as e:
                miners.dump_all()
                self.logger.error(f'Failure in operation {self.name}'
                                  f' for tcid {tcid} with'
                                  f' text:\n{text[0:3000]}...\n'
                                  f'{e}')
                with open(os.path.join(self.outpath,
                                       self.name, 'tcid_shortforms.json'),
                          'w') as f:
                    json.dump(processed, f)
                raise e
            miners.dump_all()
            with open(os.path.join(self.outpath,
                                   self.name, 'tcid_shortforms.json'),
                      'w') as f:
                json.dump(processed, f)


def mine_block(block_info):
    bucket_number, block_number, texts = block_info
    location = os.path.join(DATA_PATH, f'bucket_{bucket_number}')
    mining_operation = MiningOperation(location,
                                       f'bucket_{bucket_number}'
                                       f'-block_{block_number}',
                                       cache_size=5000)
    mining_operation(texts)


if __name__ == '__main__':
    n_workers = 63
    DATA_PATH = '/content/adeft_mining_results'
    with open(os.path.join(DATA_PATH, 'batches.json')) as f:
        buckets = json.load(f)

    for i, bucket in enumerate(buckets.values()):
        if i < 3:
            continue
        texts = get_texts(bucket)
        random.shuffle(texts)
        block_size = len(texts)//63
        blocks = [(i, j, texts[k:k+block_size])
                  for j, k in enumerate(range(0, len(texts), block_size))]
        with Pool(n_workers) as pool:
            pool.map(mine_block, blocks)
