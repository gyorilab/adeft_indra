import json
import pandas as pd

df = pd.read_csv('../../../new_grounding_table.tsv', sep='\t',
                 keep_default_na=False)
df = df.\
    groupby('grounding')['text'].\
    agg(list)

blacklist = df.to_dict()
with open('../../results/cord19_ad_blacklist.json', 'w') as f:
    json.dump(blacklist, f)
