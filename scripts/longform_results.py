import json
from adeft.discover import load_adeft_miner

with open('../data/sample_map.json') as f:
    sample_map = json.load(f)


results = {}
for index, (shortform, filename) in enumerate(list(sample_map.items())):
    if index not in [183, 185, 235, 263, 323, 364, 403, 518, 542, 581]:
        continue
    with open(f'../data/miners/{filename}') as f:
        miner = load_adeft_miner(f)
    miner.prune(2*len(shortform)+1)
    longforms = miner.get_longforms(cutoff=0.3)
    results[shortform] = longforms
    print(index)
    with open(f'../data/longform_results{index}.json', 'w') as f:
        json.dump(results, f)
    results = {}
