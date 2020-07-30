from adeft_indra.gm_mining import TfidfAgentTexts
from adeft_indra.model_building.content import get_agent_texts_for_entity

data_path = '../results/document_frequencies.json'
tfidf_background = TfidfAgentTexts(load_from=data_path)

# PKC_texts = get_agent_texts_for_entity('FPLX', 'PKC')

# tfidf_pkc = TfidfAgentTexts()
# tfidf_pkc.add(PKC_texts.values())
# h = tfidf_background.cohen_h(tfidf_pkc)

cat_texts = get_agent_texts_for_entity('FPLX', 'Cathepsin')

tfidf_cat = TfidfAgentTexts()
tfidf_cat.add(cat_texts.values())
h = tfidf_background.cohen_h(tfidf_cat)
x
