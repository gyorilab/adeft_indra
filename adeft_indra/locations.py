import os

from adeft import __version__

__all__ = ['RESOURCES_PATH', 'S3_BUCKET_ADEFT', 'S3_MODELS_PATH']

here = os.path.dirname(os.path.abspath(__file__))

# S3 bucket for storing models
S3_BUCKET_ADEFT = 'adeft'
# Path to Models folder in bucket
S3_MODELS_PATH = os.path.join(__version__, 'Models')
# S3 Bucket for resource files
S3_BUCKET_RESOURCES = 'bigmech'
# Path to document frequency dictionary in bucket
S3_DOCUMENT_FREQUENCIES_PATH = 'entrez_pubmed_dictionary.pkl'


RESOURCES_PATH = os.path.join(here, 'resources')
DOCUMENT_FREQUENCIES_PATH = os.path.join(RESOURCES_PATH,
                                         'document_frequencies.pkl')
GROUNDING_THESAURUS_PATH = os.path.join(RESOURCES_PATH,
                                        'grounding_thesaurus.json')
