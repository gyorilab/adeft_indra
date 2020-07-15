import json

from adeft.ground import AdeftGrounder

print('loading grounder')
grounder = AdeftGrounder()
print('grounder loaded')
with open('../data/longform_results_all.json') as f:
    results = json.load(f)


groundable_counts = {}
for shortform, longforms in results.items():
    rows = []
    for longform, count, score in longforms:
        grounding = grounder.ground(longform)
        if not grounding:
            grounding = None
        elif not (len(longform.split()) ==
                  len(grounding[0]['longform_text'].split())):
            grounding = None
        rows.append([longform, count, score, grounding])
    groundable_counts[shortform] = rows


g, t = 0, 0
for groundable, total, _ in groundable_counts.values():
    t += total
    g += groundable
