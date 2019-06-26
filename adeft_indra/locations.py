import os

from adeft.locations import ADEFT_PATH

from adeft import __version__

# Adeft Indra data lives inside adeft_indra folder within .adeft
ADEFT_INDRA_PATH = os.path.join(ADEFT_PATH, 'adeft_indra')
# Path to cache for text content
CACHE_PATH = os.path.join(ADEFT_INDRA_PATH, 'cache.sqlite')
# S3 Bucket for storing models
S3_MODELS_PATH = os.path.join('adeft', __version__, 'Models')


def ensure_adeft_folder():
    """Raise exception if .adeft folder does not exist on users machine."""
    if not os.path.exists(ADEFT_PATH):
        raise AdeftPathException


def ensure_adeft_indra_folder(func):
    """Create adeft_indra folder within .adeft if it does not exist."""
    def inner(*args, **kwargs):
        ensure_adeft_folder()
        if not os.path.exists(ADEFT_INDRA_PATH):
            os.mkdir(ADEFT_INDRA_PATH)
        if not os.path.exists(os.path.join(ADEFT_INDRA_PATH, 'models')):
            os.mkdir(os.path.join(ADEFT_INDRA_PATH, 'models'))
        return func(*args, **kwargs)
    return inner


class AdeftPathException(Exception):
    """Raise this exception if .adeft folder does not exist on users system"""
    def __init__(self):
        Exception.__init__(self, '.adeft folder is missing')
