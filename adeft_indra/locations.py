import os

from adeft import __version__


here = os.path.dirname(os.path.abspath(__file__))

# S3 bucket for storing models
S3_BUCKET_ADEFT = 'adeft'
# Path to Models folder in bucket
S3_MODELS_PATH = os.path.join(__version__, 'Models')
# S3 Bucket for resource files
S3_BUCKET_RESOURCES = 'bigmech'
# Path to document frequency dictionary in bucket
S3_DOCUMENT_FREQUENCIES_PATH = 'entrez_pubmed_dictionary.pkl'


ADEFT_INDRA_HOME = os.environ.get('ADEFT_INDRA_HOME') or \
    os.path.expanduser('~/.adeft_indra')


CONTENT_DB_PATH = os.path.join(ADEFT_INDRA_HOME, 'content.db')
RESULTS_DB_PATH = os.path.join(ADEFT_INDRA_HOME, 'results.db')
DOCUMENT_FREQUENCIES_PATH = os.path.join(ADEFT_INDRA_HOME,
                                         'document_frequencies.pkl')
GROUNDING_THESAURUS_PATH = os.path.join(ADEFT_INDRA_HOME,
                                        'grounding_thesaurus.json')
GROUNDER_PATH = os.path.join(ADEFT_INDRA_HOME, 'adeft_grounder.pkl')
