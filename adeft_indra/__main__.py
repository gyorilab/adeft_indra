import os
from adeft_indra.locations import ADEFT_PATH, ADEFT_INDRA_PATH

if not os.path.exists(ADEFT_PATH):
    raise RuntimeError('Adeft models path does not exist:'
                       '\nRun python -m adeft.download to create this'
                       ' folder and download pretrained models.')

if not os.path.exists(ADEFT_INDRA_PATH):
    os.mkdir(ADEFT_INDRA_PATH)
