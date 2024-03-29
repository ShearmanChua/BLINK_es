from haystack.document_stores import ElasticsearchDocumentStore

from haystack.nodes import BM25Retriever

import pandas as pd
import ast
import json 

import torch
import numpy as np
from tqdm import tqdm

wikipedia_embeddings = torch.load('/home/shearman/Desktop/work/BLINK_es/BLINK_api/models/all_entities_large.t7')
print("Number of wikipeda data embeddings: ", wikipedia_embeddings.shape[0])

json_list = []
with open('/home/shearman/Desktop/work/BLINK_es/BLINK_api/models/entity.jsonl', "r") as fin:
    lines = fin.readlines()
    for line in lines:
        entity = json.loads(line)
        json_list.append(entity)
        
print('Number of wikipedia pages: ', len(json_list))

document_store = ElasticsearchDocumentStore(host="localhost", port="9200", username="elastic", password="changeme", scheme="https", verify_certs=False, index="wikipedia",embedding_dim=wikipedia_embeddings.shape[1],search_fields=['content','title'])

retriever = BM25Retriever(document_store)

docs = []

for index in tqdm(range(0,len(json_list))):
    doc = {}
    doc['content'] = json_list[index]['text']
    doc['meta'] = {'idx':json_list[index]['idx'],'title':json_list[index]['title'], 'entity': json_list[index]['entity']}
    doc['embedding'] = wikipedia_embeddings[index].detach().cpu().numpy()

    docs.append(doc)

    if len(docs)%10000 == 0:
        document_store.write_documents(docs)
        docs = []

if len(docs) >0:
    document_store.write_documents(docs)

# uninserted_json_list_indexes = []

# for index in tqdm(range(0,len(json_list))):
#     candidate_documents = retriever.retrieve(
#         query=json_list[index]['title'],
#         top_k=1,
#         filters={"title": [json_list[index]['title']]}
#     )
#     if len(candidate_documents) > 0 and candidate_documents[0].meta['title'] == json_list[index]['title']:
#         print("Entity existed in Elasticsearch!", json_list[index]['title'])
#     else:
#         uninserted_json_list_indexes.append(index)

# with open(r'indexes.txt', 'w') as fp:
#     for item in uninserted_json_list_indexes:
#         # write each item on a new line
#         fp.write("%s\n" % item)
#     print('Done')
