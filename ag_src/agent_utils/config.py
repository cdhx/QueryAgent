# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name : config.py
   Description : config file
   Author :       HX
   date :    2023/12/3 20:03 
-------------------------------------------------
"""

import time
import sys
import pandas as pd
from agent_utils.prompt_list import *
import json
import records
PROMPT_TOO_LONG_ERROR = 'PROMPT IS TOO LONG'
config = {
    "dataset": 'grailqa',
    # "dataset": 'graphq',
    # "dataset": 'webqsp',
    # "dataset": 'metaqa',
    # "dataset": 'wikisql',
    # "golden_el": True,
    "golden_el": False,
    'model': 'gpt-3.5-turbo-0613',
    # 'self_correction': True,
    'self_correction':False,
    'query_cnt': 0,
    'openai_embedding': True,
    'sentence_transformer': False,
}
# openai embedding and sentence transformer can not be set to true at the same time
assert not (config['openai_embedding'] and config[
    'sentence_transformer']), "openai embedding and sentence transformer can not both set to True!"
# automatically set use_embedding
config['use_embedding'] = config['openai_embedding'] or config['sentence_transformer']


print(config)
null = None

note_info = ''
local_time = time.localtime(time.time())

if config['openai_embedding']:
    with open('../../data/fb_relation_embed.json', 'r', encoding='utf-8') as f:
        r_embedding_map = json.load(f)
    if config['dataset'] == 'webqsp':
        with open('../../data/WebQSP_test_question_embed.json', 'r',
                  encoding='utf-8') as f:
            q_embedding_map = json.load(f)
    elif config['dataset'] == 'grailqa':
        with open('../../data/grailqa_question_embed.json', 'r',
                  encoding='utf-8') as f:
            q_embedding_map = json.load(f)
    elif config['dataset'] == 'graphq':
        with open('../../data/graphq_question_embed.json', 'r',
                  encoding='utf-8') as f:
            q_embedding_map = json.load(f)


result_file_name = "../../logs/"
if config['dataset'] == 'grailqa':
    data_path = "../../data/GrailQA_v1.0/grailqa_dev_pyql.json"
    linking_path = '../../data/GrailQA_v1.0/grail_pangu_tiara.json'
    DATASET_PROMPT = GRILQA_PROMPT
elif config['dataset'] == 'graphq':
    data_path = "../../data/GraphQ/graphquestions_v1_fb15_test_091420.json"
    linking_path = "../../data/GraphQ/graphq_test.json"
    DATASET_PROMPT = GRAPHQ_PROMPT
elif config['dataset'] == 'metaqa':
    data_path = "../../data/metaQA/qa_test_3hop.txt"
    DATASET_PROMPT = METAQA_3HOP_PROMPT
elif config['dataset'] == 'webqsp':
    data_path = "../../data/WebQSP/WebQSP_test_processed.json"
    linking_path = '../../data/WebQSP/webqsp_test_el.json'
    DATASET_PROMPT = WEBQSP_PROMPT
elif config['dataset'] == 'wikisql':
    data_path = "../../data/wikisql/wikisql_test.json"
    db = records.Database('sqlite:///../../data/wikisql/test.db')
    conn = db.get_connection()
    DATASET_PROMPT = WIKISQL_PROMPT
else:
    raise ValueError("Invalid dataset, valid dataset include: grailqa,webqsp, graphq and metaqa, while your dataset is ",
                     config['dataset'])


if config['dataset'] == 'grailqa':
    SPARQLPATH = "http://114.212.81.217:8896/sparql"
elif config['dataset'] == 'graphq':
    SPARQLPATH = "http://114.212.81.217:8896/sparql"
elif config['dataset'] == 'metaqa':
    SPARQLPATH = "http://114.212.81.217:8896/sparql"
elif config['dataset'] == 'webqsp':
    SPARQLPATH = "http://114.212.81.217:8896/sparql"
elif config['dataset'] == 'wikisql':
    data_path = "../../data/wikisql/wikisql_test.json"
    db = records.Database('sqlite:///../../data/wikisql/test.db')
    conn = db.get_connection()
else:
    raise ValueError("Invalid dataset, valid dataset include: grailqa, graphq,webqsp and wikisql, while your dataset is ",
                     config['dataset'])


result_file_name += (config['dataset'] + '_' + config['model'] + '_' + (
    'sc' if config['self_correction'] else 'direct') + '_' + ('gold' if config['golden_el'] else  'el') + '_' + (
                         'openai_emb' if config['openai_embedding'] else 'sent_emb' if config[
                             'sentence_transformer'] else 'bm25') +
                     '_' + ('0' if len(str(local_time.tm_mon)) == 1 else '') + str(local_time.tm_mon) +
                     '_' + ('0' if len(str(local_time.tm_mday)) == 1 else '') + str(local_time.tm_mday) +
                     '_' + ('0' if len(str(local_time.tm_hour)) == 1 else '') + str(local_time.tm_hour) +
                     '_' + ('0' if len(str(local_time.tm_min)) == 1 else '') + str(local_time.tm_min) +
                     '_' + ('0' if len(str(local_time.tm_sec)) == 1 else '') + str(local_time.tm_sec)) + (
                        '_' + note_info if note_info != '' else "")

import random
all_key=['YOU_API_KEY']
random.shuffle(all_key)
