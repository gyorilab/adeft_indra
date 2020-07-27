import boto3


import adeft_indra.locations as loc


def download_DF_dictionary():
    client = boto3.client('s3')
    client.download_file(loc.S3_BUCKET_RESOURCES,
                         loc.S3_DOCUMENT_FREQUENCIES_PATH,
                         loc.S3_DOCUMENT_FREQUENCIES_PATH)


if __name__ == '__main__':
    download_DF_dictionary()
