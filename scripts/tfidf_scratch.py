import json
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from gensim.models import TfidfModel
from gensim.sklearn_api.tfidf import TfIdfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from indra.literature.adeft_tools import universal_extract_text
from indra_db.util.content_scripts import get_stmts_with_agent_text_like, \
    get_text_content_from_stmt_ids

tokenize = TfidfVectorizer().build_tokenizer()


def preprocess(text):
    return [token.lower() for token in tokenize(text)]


ER_stmts = get_stmts_with_agent_text_like('ER')['ER']
refs, content = get_text_content_from_stmt_ids(ER_stmts)

texts = [universal_extract_text(text) for text in content.values() if text]

processed_texts = [preprocess(text) for text in texts]
dct = Dictionary(processed_texts)
model = TfidfModel(dictionary=dct)

corpus = [dct.doc2bow(line) for line in processed_texts]

example = preprocess("This is an example text that isn't about ER.")
a = dct.doc2bow(example)
rep = model[a]


AA_stmts = get_stmts_with_agent_text_like('AA')['AA']
AA_refs, AA_content = get_text_content_from_stmt_ids(AA_stmts)
AA_texts = [universal_extract_text(text)
            for text in AA_content.values() if text]
processed_AA_texts = [preprocess(text) for text in AA_texts]
AA_dct = Dictionary(processed_AA_texts)
AA_dct.filter_extremes(no_below=1, no_above=1.0, keep_n=1000)

token2id = model.id2word.token2id
id_mapping = {key: token2id[value] for key, value in AA_dct.items()}
AA_idfs = {key: model.idfs[value] for key, value in id_mapping.items()}

with open('../data/entrez_all_pmids.json', 'r') as f:
    pmids = json.load(f)
