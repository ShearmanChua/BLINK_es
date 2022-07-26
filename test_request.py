import requests
import json
from typing import List, Dict
import pandas as pd
import ast
import re
import time

from haystack.document_stores import ElasticsearchDocumentStore

def predict_jerex(dataset: Dict):
    response = requests.post('http://0.0.0.0:8080/df_link', json = dataset)
    
    #ipdb.set_trace()
    # print([doc for doc in response.iter_lines()])
    response = response.json()
    # print(type(response))
    # df_json = json.loads(response)
    # print(type(df_json))
    df = pd.json_normalize(response, max_level=0)

    print(df.head())
    print(df.info())

    df.to_csv("data/test_jerex.csv",index=False)

    return df

def predict_blink(dataset: Dict):

    response = requests.post('http://0.0.0.0:5050/df_link', json = dataset)
    # df_json = json.dumps(response.json())
    # df = pd.read_json(df_json, orient="records")

    # print(df.head())
    # print(df.info())

    df = pd.json_normalize(response.json(), max_level=0)

    print(df)
    print(df.info())

    df.to_csv("data/articles_entity_linked.csv",index=False)

    return df

def generate_entity_linking_df(results_df):

    entities_linking_df = pd.DataFrame(columns=['doc_id','mention', 'mention_type','context_left','context_right'])

    for idx, row in results_df.iterrows():
        doc_id = row['doc_id']
        if type(row['relations']) == str:
            relations = ast.literal_eval(row['relations'])
            tokens = ast.literal_eval(row['tokens'])
        else:
            relations =  row['relations']
            tokens = row['tokens']
        entities = []
        for relation in relations:
            head_entity = " ".join(tokens[relation['head_span'][0]:relation['head_span'][1]])
            if head_entity not in entities:
                print("Head Entity:")
                print(head_entity)
                left_context = " ".join(tokens[relation['head_span'][0]-100:relation['head_span'][0]])
                left_context = re.sub(r"\S*https?:\S*", '', left_context)
                print("Left context: ",left_context)
                print("\n")
                right_context = " ".join(tokens[relation['head_span'][1]:relation['head_span'][1]+100])
                right_context = re.sub(r"\S*https?:\S*", '', right_context)
                print("Right context: ",right_context)
                print("\n")
                entities_linking_df.loc[-1] = [doc_id, head_entity, relation['head_type'], left_context, right_context]  # adding a row
                entities_linking_df.index = entities_linking_df.index + 1  # shifting index
                entities_linking_df = entities_linking_df.sort_index()  # sorting by index
                entities.append(head_entity)

            tail_entity = " ".join(tokens[relation['tail_span'][0]:relation['tail_span'][1]])
            if tail_entity not in entities:
                print("Tail Entity:")
                print(tail_entity)
                left_context = " ".join(tokens[relation['tail_span'][0]-100:relation['tail_span'][0]])
                left_context = re.sub(r"\S*https?:\S*", '', left_context)
                print("Left context: ",left_context)
                print("\n")
                right_context = " ".join(tokens[relation['tail_span'][1]:relation['tail_span'][1]+100])
                right_context = re.sub(r"\S*https?:\S*", '', right_context)
                print("Right context: ",right_context)
                print("\n")
                entities_linking_df.loc[-1] = [doc_id, tail_entity, relation['tail_type'], left_context, right_context]  # adding a row
                entities_linking_df.index = entities_linking_df.index + 1  # shifting index
                entities_linking_df = entities_linking_df.sort_index()  # sorting by index
                entities.append(tail_entity)

    print(entities_linking_df.head())
    return entities_linking_df


if __name__ == '__main__':

    start = time.time()

    document_store = ElasticsearchDocumentStore(host= "localhost",
                                                port= "9200", 
                                                username= "elastic", 
                                                password= "changeme", 
                                                scheme= "https", 
                                                verify_certs= False, 
                                                index = 'formula1_articles',
                                                search_fields= ['content','title'])

    documents = document_store.get_all_documents()

    articles_df = pd.DataFrame(columns=['ID','title','text','elasticsearch_ID'])

    for document in documents:
        articles_df.loc[-1] = [document.meta['ID'], document.meta['link'],document.content,document.id]  # adding a row
        articles_df.index = articles_df.index + 1  # shifting index
        articles_df = articles_df.sort_index()  # sorting by index

    print(articles_df.info())
    print(articles_df.head())

    # df_json = articles_df.to_json(orient="records")
    # df_json = json.loads(df_json)
    # jerex_results = predict_jerex(df_json)
    # print("jerex results: ", jerex_results)

    # jerex_results = pd.read_csv('data/test_jerex.csv')

    # entity_linking_df = generate_entity_linking_df(jerex_results)

    # print(entity_linking_df)
    # entity_linking_df.to_csv("data/entity_linking_df.csv",index=False)
    # entity_linking_df = pd.read_csv('/home/shearman/Desktop/work/BLINK_es/data/entity_linking_df.csv')
    # entity_linking_df =entity_linking_df.iloc[:10,:]


    # df_json = entity_linking_df.to_json(orient="records")
    # df_json = json.loads(df_json)

    # blink_results = predict_blink(df_json)

    df = pd.read_csv('data/articles_entity_linked.csv')

    list_of_cluster_dfs = df.groupby('doc_id')

    entities = []
    ids = []

    for group, cluster_df in list_of_cluster_dfs:
        doc_entities = []
        doc_id = cluster_df['doc_id'].tolist()[0]
        mentions = cluster_df['mention'].tolist()
        mentions_type = cluster_df['mention_type'].tolist()
        entity_links = cluster_df['entity_link'].tolist()
        entity_names = cluster_df['entity_names'].tolist()
        for idx in range(0,len(mentions)):
            mention = dict()
            mention['mention'] = mentions[idx]
            mention['mention_type'] = mentions_type[idx]
            mention['entity_link'] = entity_links[idx]
            mention['entity_name'] = entity_names[idx]
            doc_entities.append(mention)
        ids.append(doc_id)
        entities.append(doc_entities)

    entities_df = pd.DataFrame()
    entities_df['ID'] = ids
    entities_df['identified_entities'] = entities

    results_df = pd.merge(articles_df, entities_df, on=["ID"])

    print(results_df.info())    

    for idx, row in results_df.iterrows():
        meta_dict = {'entities_identified':row['identified_entities']}
        document_store.update_document_meta(id=row['elasticsearch_ID'], meta=meta_dict)

    end = time.time()
    print("Time to complete jerex and entity linking",end - start)

