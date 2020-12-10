from indra_db.util.content_scripts import get_agent_stmt_counts

from adeft_indra.gm_mining import TfidfAgentTexts
from adeft_indra.db.content import get_agent_texts_for_entity


data_path = '../results/document_frequencies.json'
tfidf_background = TfidfAgentTexts(load_from=data_path)


def get_gm_mining_table(entity, top=200):
    texts = get_agent_texts_for_entity('FPLX', entity)
    tfidf = TfidfAgentTexts()
    tfidf.add(texts.values())
    h = tfidf_background.cohen_h(tfidf)
    h = h[:200]
    agent_texts = [text for text, score in h]
    count_dict = get_agent_stmt_counts(agent_texts)
    rows = [[text, score, count_dict[text]] for text, score in h]
    return sorted(rows, key=lambda x: -x[1])
