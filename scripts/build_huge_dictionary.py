import json
from itertools import zip_longest

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.feature_extraction.text import TfidfVectorizer
from indra.literature.adeft_tools import universal_extract_text
from indra_db.util.content_scripts import get_text_content_from_pmids


tokenize = TfidfVectorizer().build_tokenizer()


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def preprocess(text):
    return [token.lower() for token in tokenize(text)]


with open('../data/entrez_all_pmids.json') as f:
    pmids = json.load(f)

all_pmids = set()
for pmid_list in pmids.values():
    all_pmids |= set(pmid_list)


class ContentIterator(object):
    def __init__(self, pmid_list, chunksize=10000):
        self.pmids = pmid_list
        self.chunksize = chunksize

    def __iter__(self):
        groups = grouper(self.pmids, self.chunksize)
        for pmids in groups:
            _, content = get_text_content_from_pmids(pmids)
            texts = [universal_extract_text(text)
                     for text in content.values() if text]
            for text in texts:
                yield preprocess(text)


all_pubmed_content = ContentIterator(all_pmids)
dictionary = Dictionary(text for text in all_pubmed_content)
location = '../results/pubmed_dictionary.pkl'
dictionary.save(location)

dictionary = Dictionary.load(location)


token_dfs = {key: dictionary.dfs[value]
             for key, value in dictionary.token2id.items()}

dictionary.filter_extremes(no_below=3, no_above=0.25, keep_n=None)
location_filtered = '../results/pubmed_dictionary_filtered.pkl'
dictionary.save(location_filtered)

model = TfidfModel(dictionary=dictionary)
