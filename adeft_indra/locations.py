import os

from adeft import __version__


here = os.path.dirname(os.path.abspath(__file__))

# S3 bucket for storing models
S3_BUCKET_ADEFT = 'adeft'
# Path to Models folder in bucket
S3_MODELS_PATH = os.path.join(__version__, 'Models')


ADEFT_INDRA_HOME = os.environ.get('ADEFT_INDRA_HOME') or \
    os.path.expanduser('~/.adeft_indra')

GROUNDING_THESAURUS_PATH = os.path.join(ADEFT_INDRA_HOME,
                                        'grounding_thesaurus.json')
GROUNDER_PATH = os.path.join(ADEFT_INDRA_HOME, 'adeft_grounder.pkl')
