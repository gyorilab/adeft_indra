import os
import json

from indra.literature.adeft_tools import get_text_content_for_pmids

from adeft_indra.s3 import escape_filename

with open('entrez_all_pmids.json') as f:
    pmid_map = json.load(f)

content_directory = os.path.join('..', 'entrez_content')
if not os.path.exists(content_directory):
    os.makedirs(content_directory)

for gene, pmids in pmid_map.items():
    content = get_text_content_for_pmids(pmids)
    with open(os.path.join(content_directory,
                           f'{escape_filename(gene)}_content.json'),
              'w') as f:
        json.dump(content, f)
