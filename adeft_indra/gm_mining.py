import os
import json
from collections import Counter
from adeft_indra.model_building.content import get_pmids_for_entity
from indra_db.util.content_scripts import get_agent_texts_for_pmids


def get_agent_text_dfs(count_dicts):
    dfs = Counter()
    for count_dict in count_dicts:
        text_set = set()
        for agent_text, _ in count_dict.items():
            text_set.add(agent_text)
        for agent_text in text_set:
            dfs[agent_text] += 1
    return dfs


class TfidfAgentTexts(object):
    def __init__(self, load_from=None, n_jobs=1):
        if load_from is None:
            DFs = Counter()
            total_documents = 0
        else:
            with open(os.path.realpath(os.path.expanduser(load_from))) as f:
                data = json.load(f)
                DFs, total_documents = data['DFs'], data['total_documents']
        self.DFs = DFs
        self.total_documents = total_documents
        self.n_jobs = n_jobs

    def add(self, count_dicts):
        for count_dict in count_dicts:
            text_set = set()
            for agent_text, _ in count_dict.items():
                text_set.add(agent_text)
            for agent_text in text_set:
                self.DFs[agent_text] += 1
            self.total_documents += 1

    def dump(self, filepath):
        filepath = os.path.realpath(os.path.expanduser(filepath))
        with open(filepath, 'w') as f:
            json.dump({'DFs': dict(self.Dfs),
                       'total_documents': self.total_documents}, f)


pmids = get_pmids_for_entity('FPLX', 'Carboxylesterase')
texts = get_agent_texts_for_pmids(pmids)

tfidf = TfidfAgentTexts()
tfidf.add(texts.values())
