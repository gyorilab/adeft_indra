import json
import boto3
import pandas as pd

from gilda.resources import GROUNDING_TERMS_PATH


import adeft_indra.locations as loc


def download_DF_dictionary():
    client = boto3.client('s3')
    client.download_file(loc.S3_BUCKET_RESOURCES,
                         loc.S3_DOCUMENT_FREQUENCIES_PATH,
                         loc.S3_DOCUMENT_FREQUENCIES_PATH)


def generate_grounding_thesaurus():
    gt = pd.read_csv(GROUNDING_TERMS_PATH, sep='\t',
                     names=['normtext', 'text', 'namespace', 'id',
                            'standard_name', 'type', 'source'])
    gt = gt[['text', 'namespace', 'id']]
    gt['grounding'] = gt.apply(lambda row: f'{row.namespace}:{row.id}',
                               axis=1)
    gt = gt[['text', 'grounding']].\
        groupby('grounding', as_index=False).agg(list)
    thesaurus = {}
    for _, row in gt.iterrows():
        thesaurus[row.grounding] = row.text
    with open(loc.GROUNDING_THESAURUS_PATH, 'w') as f:
        json.dump(thesaurus, f, indent=True)


if __name__ == '__main__':
    download_DF_dictionary()
    generate_grounding_thesaurus()
