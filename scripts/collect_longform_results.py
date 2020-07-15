import json

results = {}
for i in range(631):
    try:
        with open(f'../data/longform_results{i}.json') as f:
            results.update(json.load(f))
    except Exception:
        continue
