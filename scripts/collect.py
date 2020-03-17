import os
import json
import random
import logging
import multiprocessing as mp
from adeft.discover import load_adeft_miner


merged_path = '/content/adeft_mining_results/merged_results'
logger = logging.getLogger('collect')
file_handler = logging.FileHandler(os.path.join(merged_path,
                                                'collect.log'))
log_format = logging.Formatter('%(asctime)s - %(name)s - '
                               '%(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

manager = mp.Manager()
filenames_map = manager.dict({})


def get_shortform(names):
    result = {}
    for name in names:
        try:
            with open(os.path.join(merged_path, 'miners', name)) as f:
                miner = load_adeft_miner(f)
            shortform = miner.shortform
            result[shortform] = name
        except Exception:
            logger.warning(f'unable to process file {name}')
            pass
    for key, value in result.items():
        filenames_map[key] = value


filenames = os.listdir(os.path.join(merged_path, 'miners'))
random.shuffle(filenames)
blocks = [filenames[i:i+6000] for i in range(0, len(filenames), 6000)]

with mp.Pool(64) as pool:
    pool.map(get_shortform, blocks)


with open(os.path.join(merged_path, 'filenames.json'), 'w') as f:
    json.dump(dict(filenames_map), f)
