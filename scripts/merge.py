import os
import json
import uuid
import logging
import multiprocessing as mp
from collections import defaultdict

from adeft.discover import compose, load_adeft_miner


def build_filename_mapping():
    result = defaultdict(list)
    for i in range(26):
        directory = f'/content/adeft_mining_results/bucket_{i}'
        for block in os.listdir(directory):
            if not os.path.isdir(os.path.join(directory, block)):
                continue
            with open(os.path.join(directory, block, 'miners',
                                   'filenames.json')) as f:
                filenames_map = json.load(f)
            for key, value in filenames_map.items():
                result[key].append(os.path.join(directory, block,
                                                'miners', value))
    return result


merged_path = os.path.join('/content', 'adeft_mining_results',
                           'merged_results')

with open(os.path.join(merged_path, 'big_filenames_map.json')) as f:
    big_filenames_map = json.load(f)

logger = logging.getLogger('merge')
file_handler = logging.FileHandler(os.path.join(merged_path, 'merge.log'))
log_format = logging.Formatter('%(asctime)s - %(name)s - '
                               '%(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

lock = mp.Lock()
filenames = {}


def merge_results(shortform):
    try:
        filenames = big_filenames_map[shortform]
        miners = []
        for filename in filenames:
            with open(filename, 'r') as f:
                miner = load_adeft_miner(f)
                miners.append(miner)
        result = compose(*miners)
        outfile = uuid.uuid1().hex
        with open(os.path.join(merged_path, 'miners', outfile), 'w') as f:
            result.dump(f)
        return (shortform, outfile)
    except Exception:
        return (shortform, None)


if __name__ == '__main__':
    pool = mp.Pool(64)
    res = pool.map(merge_results, big_filenames_map.keys())
    filenames = dict(res)
    with open(os.path.join(merged_path, 'filenames.json'), 'w') as f:
        json.dump(filenames, f)
