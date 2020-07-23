import os
import json
import boto3
import tempfile

from adeft.download import get_s3_models

from adeft_indra.locations import S3_BUCKET, S3_MODELS_PATH


def model_to_s3(disambiguator):
    grounding_dict = disambiguator.grounding_dict
    names = disambiguator.names
    classifier = disambiguator.classifier

    shortforms = disambiguator.shortforms
    model_name = escape_filename(':'.join(sorted(escape_filename(shortform)
                                                 for shortform in shortforms)))

    model_map = {key: model_name for key in grounding_dict}
    s3_models = get_s3_models()
    s3_models.update(model_map)
    client = boto3.client('s3')
    with tempfile.NamedTemporaryFile() as temp:
        with open(temp.name, 'w') as f:
            json.dump(s3_models, f)
        client.upload_file(temp.name, S3_BUCKET,
                           os.path.join(S3_MODELS_PATH, 's3_models.json'))

    with tempfile.NamedTemporaryFile() as temp:
        with open(temp.name, 'w') as f:
            json.dump(grounding_dict, f)
        client.upload_file(temp.name, S3_BUCKET,
                           os.path.join(S3_MODELS_PATH, model_name,
                                        f'{model_name}_grounding_dict.json'))

    with tempfile.NamedTemporaryFile() as temp:
        with open(temp.name, 'w') as f:
            json.dump(names, f)
        client.upload_file(temp.name, S3_BUCKET,
                           os.path.join(S3_MODELS_PATH, model_name,
                                        f'{model_name}_names.json'))

    with tempfile.NamedTemporaryFile() as temp:
        classifier.dump_model(temp.name)
        client.upload_file(temp.name, S3_BUCKET,
                           os.path.join(S3_MODELS_PATH, model_name,
                                        f'{model_name}_model.gz'))


_escape_map = {'_': '_',
               '/': 's'}


def _escape(char):
    if char in _escape_map:
        return '_' + _escape_map[char]
    if char.islower():
        return '_' + char.upper()
    else:
        return char


def escape_filename(filename):
    """Convert filename for one with escape character before lowercase

    This is done to handle case insensitive file systems. _ is used as an
    escape character. It is also an escape character for itself.
    """
    return ''.join([_escape(char) for char in filename])


def unescape_filename(filename):
    """Inverse of escape_filename"""
    unescape_map = {value: key for key, value in _escape_map.items()}
    escape = False
    output = []
    for char in filename:
        if escape:
            if char in unescape_map:
                output.append(unescape_map[char])
            else:
                output.append(char.lower())
            escape = False
        elif char == '_':
            escape = True
        elif char in _escape_map:
            raise ValueError(f'Filename {filename} contains invalid'
                             ' characters')
        else:
            output.append(char)
    return ''.join(output)
