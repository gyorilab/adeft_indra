import os

from adeft.locations import ADEFT_PATH

from adeft_indra.locations import ADEFT_INDRA_PATH


def ensure_adeft_folder():
    """Raise exception if .adeft folder does not exist on users machine."""
    if not os.path.exists(ADEFT_PATH):
        raise AdeftPathException('.adeft folder is missing')


def create_adeft_indra_folder():
    """Create adeft_indra folder within .adeft if it does not exist."""
    ensure_adeft_folder()
    if not os.path.exists(ADEFT_INDRA_PATH):
        os.mkdir(ADEFT_INDRA_PATH)


class AdeftPathException(Exception):
    """Raise this exception if .adeft folder is missing."""
    pass
