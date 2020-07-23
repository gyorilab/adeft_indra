import os

from adeft import __version__


# S3 Bucket for storing models
S3_BUCKET_ADEFT = 'adeft'
S3_MODELS_PATH = os.path.join(__version__, 'Models')

