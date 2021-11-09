import json
from itertools import zip_longest

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.feature_extraction.text import TfidfVectorizer

from adeft_indra.content import get_plaintexts_for_pmids


tokenize = TfidfVectorizer().build_tokenizer()


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def preprocess(text):
    return [token.lower() for token in tokenize(text)]


with open('../data/combined_pmids.json') as f:
    all_pmids = json.load(f)


class ContentIterator(object):
    def __init__(self, pmid_list, chunksize=10000):
        self.pmids = pmid_list
        self.chunksize = chunksize

    def __iter__(self):
        groups = grouper(self.pmids, self.chunksize)
        for pmids in groups:
            texts = get_plaintexts_for_pmids(pmids).values()
            for text in texts:
                yield preprocess(text)


all_pubmed_content = ContentIterator(all_pmids)
dictionary = Dictionary((text for text in all_pubmed_content), prune_at=None)
location = '../results/pubmed_dictionary.pkl'
dictionary.save(location)
