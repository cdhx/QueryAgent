# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name : ag_utils.py 
   Description :  utils func
   Author :       HX
   date :    2023/11/20 21:38 
-------------------------------------------------
"""
from SPARQLWrapper import SPARQLWrapper, JSON
import os
import re
import json
import pandas as pd
import numpy as np
from agent_utils.config import *
from tqdm import tqdm
import tiktoken
import openai
import requests

sparql_id_fb = """PREFIX : <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  ?entity :type.object.name ?tailEntity .\n    FILTER(?entity = :%s)  \n}"""

def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def clean_para(para):
    # Remove redundant quotes
    if type(para) == str:
        para = para.strip()
        if len(para) != 0:
            # If it's a literal, the first one should be in quotes, and there should be quotes inside the literal.
            if 'XMLSchema#' in para:
                if para[-1] != '"':
                    para = '"' + para[1:]
                if para[-1] == '"' or para[-1] == "'":
                    para = para[:-1]
            else:
                if para[0] == '"' or para[0] == "'":
                    para = para[1:]
                if para[-1] == '"' or para[-1] == "'":
                    para = para[:-1]
    return para

def num_tokens_from_string(string: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def readjson(path):
    with open(path, mode='r', encoding='utf-8') as load_f:
        data_ = json.load(load_f)
    return data_


class Dict2Obj(dict):
    def __getattr__(self, key):
        value = self.get(key)
        return Dict(value) if isinstance(value, dict) else value

    def __setattr__(self, key, value):
        self[key] = value


class IntEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, int):
            return int(obj)
        return super().default(obj)


def savejson(file_name, json_info, indent=4):
    with open('{}.json'.format(file_name), 'w') as fp:
        json.dump(json_info, fp, indent=indent, sort_keys=False, cls=IntEncoder)


def print_rgb(r, g, b, input_str):
    # Setting custom output colors
    custom_color = '\033[38;2;' + str(r) + ';' + str(g) + ';' + str(b) + 'm' + input_str + '\033[0m'
    print(custom_color)


def print_thought(*input_str):
    # blue for thought
    print_rgb(82, 121, 186, ' '.join(input_str))


def print_action(*input_str):
    # green for action
    print_rgb(90, 187, 147, ' '.join(input_str))


def print_obs(*input_str):
    # purple for observation
    print_rgb(141, 98, 157, ' '.join(input_str))


def print_error(*input_str):
    # red for error
    print_rgb(255, 105, 97, ' '.join(input_str))


def print_refine(*input_str):
    # yellow for refine
    print_rgb(255, 200, 0, ' '.join(input_str))


def f1_score(pred, golden):
    if len(pred) == 0 and len(golden) == 0:
        f1 = 1
    elif len(pred) == 0 and len(golden) != 0:
        f1 = 0
    elif len(pred) != 0 and len(golden) == 0:
        f1 = 0
    else:
        p = len([x for x in pred if x in golden]) / len(pred)
        r = len([x for x in golden if x in pred]) / len(golden)
        if p == 0 or r == 0:
            f1 = 0
        else:
            f1 = 2 * p * r / (p + r)
    return f1


def llm(prompt, model_name, stop=["\n"]):
    got_result = False
    while got_result != True:
        try:
            current_key=all_key[0]
            del all_key[0]
            all_key.append(current_key)

            openai.api_key = current_key

            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop
            )


            response = response["choices"][0]['message']['content'].strip()
            got_result = True

        except Exception as e:
            if 'You exceeded your current quota' in str(e) or 'The OpenAI account associated with this API key' in str(e):
                pass
                print('bad key: ',current_key)
            # prompt too long
            elif "This model's maximum context length is" in str(e) or "Request too large for" in str(e):
                print(e)
                return PROMPT_TOO_LONG_ERROR
            else:
                print(e)


    return response


def abandon_rels(relation):
    # abandon bad relation
    if config['dataset'] == 'webqsp':

        if (relation == "type.object.type" or relation == "type.object.name"
                or relation.startswith("common.") or relation.startswith("freebase.")
                or "sameAs" in relation):

            return True
    elif config['dataset'] == 'grailqa':

        if (relation.startswith("type.") or relation.startswith("common.") or relation.startswith("freebase.")):
            return True
    elif config['dataset'] == 'graphq':
        if (relation == "type.object.type" or relation == "type.object.name"
         or relation.startswith("common.") or relation.startswith("freebase.")
         or "sameAs" in relation):
            return True

def replace_relation_prefix(relations):
    return [relation['relation'].replace("http://rdf.freebase.com/ns/", "") for relation in relations]


def replace_entities_prefix(entities):
    return [entity['tailEntity'].replace("http://rdf.freebase.com/ns/", "") for entity in entities]


def table_result_to_list(res):
    #  transform this format [{'p': 'http://rdf.freebase.com/ns/common.topic.image',
    #   's': 'http://rdf.freebase.com/ns/m.0crkzcy'},
    #  {'p': 'http://rdf.freebase.com/ns/meteorology.tropical_cyclone.tropical_cyclone_season',
    #   's': 'http://rdf.freebase.com/ns/m.06tgzm'}]
    # to this format
    # {'p': ['http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
    #   'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',]
    #  's': ['http://rdf.freebase.com/ns/common.topic',
    #   'http://rdf.freebase.com/ns/common.topic']}
    if len(res) == 0:
        return {}
    else:
        key_list = res[0].keys()
        result = {}
        for key in key_list:
            result[key] = list(set([item[key] for item in res]))
        return result


def execute_sql(sql_txt):
    res = conn.query(sql_txt).all()
    res = [x.as_dict() for x in res]
    return res


def execute_sparql(sparql_txt):
    # execute SPARQL
    config['query_cnt'] += 1
    if config['dataset'] in ['grailqa', 'webqsp', 'graphq']:
        sparql_txt = 'PREFIX : <http://rdf.freebase.com/ns/>\n' + sparql_txt

    try:
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        res = []
        for x in results["results"]["bindings"]:
            res_item = {}
            for k, v in x.items():
                res_item[k] = v['value']
            res.append(res_item)
        return res
    except:
        print("SPARQL query error")
        print(sparql_txt)
        return []

def id2label(entity_id):
    # query the label of given ID
    valid_id = False
    if config['dataset'] in ['grailqa', 'webqsp', 'graphq'] and entity_id.startswith('m.'):
        sparql_query = sparql_id_fb % (entity_id)
        valid_id = True
    if valid_id:
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        if len(results["results"]["bindings"]) == 0:
            return "UnName_Entity"
        else:
            # return the first english labeol
            for lines in results["results"]["bindings"]:
                if lines['tailEntity']['xml:lang'] == 'en':
                    return lines['tailEntity']['value']
            # if there is no en label, just return the first one
            return results["results"]["bindings"][0]['tailEntity']['value']
    else:
        return entity_id


def get_brief_obs(obs):
    # get a brief version of the observation
    brief_obs = obs
    # for get_relaion
    if type(obs) == dict:
        brief_obs = {}
        for k, v in obs.items():
            if type(v) == list:
                brief_obs[k] = v[:3] + ['...'] if len(v) > 3 else v
            elif type(v) == dict:
                brief_obs[k] = dict(list(v.items())[:3]) if len(v) > 3 else v

    elif type(obs) == str:
        #  for query result
        if "The value of variable" in obs:
            match_res = re.findall('The value of variable (\?.*?) is (\[.*?\])', obs)
            brief_obs = ""
            for tur in match_res:
                res_list = eval(tur[1])
                if len(res_list) <= 2:
                    brief_obs += 'The value of variable ' + tur[0] + ' is ' + tur[1] + '. '
                else:
                    brief_obs += 'The value of variable ' + tur[0] + ' is ' + str(eval(tur[1])[:2] + (['...'])) + '. '

    return str(brief_obs)



def get_dynamic_history(solve_history):
    # Dynamically change the prompt, forget too long ago observation
    if config['dataset'] == 'metaqa':
        dynamic_history = solve_history['base_prompt'] + "\nQuestion: " + solve_history[
            'question'] + " \nEntity: " + str(solve_history['entity']) + "\nRelation for the entity: " + str(
            solve_history['initial_rel']) + "\n"
    elif config['dataset'] == 'wikisql':
        dynamic_history = solve_history['base_prompt'] + "\nQuestion: " + solve_history[
            'question'] + " \nTable Header: " + str(solve_history['header']) + "\n"
    elif config['dataset'] == 'wtq':
        dynamic_history = solve_history['base_prompt'] + "\nQuestion: " + solve_history[
            'question'] + " \nTable Header: " + str(solve_history['header']) + "\n"
    else:
        dynamic_history = solve_history['base_prompt'] + "\nQuestion: " + solve_history[
            'question'] + " \nEntity: " + str(solve_history['entity']) + "\n"

    for index, his in enumerate(solve_history['TAO_list']):
        # if it's two steps ago, forget the specifics of observation
        if index < len(solve_history['TAO_list']) - 2:
            obs = "It's not important. Focus on recent observations."
            dynamic_history += f"Thought {index + 1}: {his['Thought']}\nAction {index + 1}: {his['Action']}\nObservation {index + 1}: {obs}\n"
        elif index == len(solve_history['TAO_list']) - 2:
            obs = str(his['Observation'])
            if 'ERROR_IN_STEP' in obs and len(obs) > 250:
                obs = "It's not important. Focus on recent observations."
            dynamic_history += f"Thought {index + 1}: {his['Thought']}\nAction {index + 1}: {his['Action']}\nObservation {index + 1}: {obs}\n"
        else:
            dynamic_history += f"Thought {index + 1}: {his['Thought']}\nAction {index + 1}: {his['Action']}\nObservation {index + 1}: {str(his['Observation'])}\n"

    dynamic_history = dynamic_history + f"Thought {len(solve_history['TAO_list']) + 1}:"
    return dynamic_history


def try_step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1



def process_webqsp():
    pyql_file = readjson('../../data/webqsp_test_pyql.json')
    webqsp_train = readjson('/home2/xhuang/PycharmProject/GLLLM/ReAct/data/WebQSP/WebQSP.test.json')['Questions']
    processed_examples = list()

    for example in tqdm(webqsp_train):
        # follow GMT-KBQA
        parse = example["Parses"][0]
        # 1. topic entity
        grounded_items = {}
        if "TopicEntityMid" in parse and parse["TopicEntityMid"]:
            grounded_items.update({
                parse["TopicEntityName"]: parse["TopicEntityMid"]
            })
        # 2. constrain
        if "Constraints" in parse:
            for cons in parse["Constraints"]:
                if cons["ArgumentType"] == "Entity":
                    grounded_items.update({
                        cons["EntityName"]: cons["Argument"],
                    })
                elif cons["ArgumentType"] == "Value":
                    if cons["ValueType"] == "String":
                        mid = f'"{cons["Argument"]}"'
                        grounded_items.update({
                            mid: f'{mid}@en'
                        })
                    elif cons["ValueType"] == "DateTime":
                        mid = f'"{cons["Argument"]}"^^<http://www.w3.org/2001/XMLSchema#dateTime>'
                        grounded_items.update({
                            cons["Argument"]: mid,
                        })
                    else:
                        raise Exception(f'cons: {cons}; example id: {example["QuestionId"]}')
                else:
                    raise Exception(f'cons: {cons}; example id: {example["QuestionId"]}')

        # answer of different parser
        different_answer = []
        for par in example['Parses']:
            exe_res = table_result_to_list(execute_sparql(par['Sparql']))
            if exe_res == []:
                exe_res = []
            else:
                if len(exe_res.keys()) > 1:
                    print(example['QuestionId'], ' multi answer key')

                answer_key = list(exe_res.keys())[0]
                exe_res = exe_res[answer_key]
                exe_res = [x.replace("http://rdf.freebase.com/ns/", "") for x in exe_res]
            different_answer.append(exe_res)

            exe_res.sort()
            ori_res = [x['AnswerArgument'] for x in par['Answers']]
            ori_res.sort()
            if ori_res != exe_res:
                print(example['QuestionId'])
                print(ori_res)
                print(exe_res)

        processed_examples.append({
            "qid": example["QuestionId"],
            "question": example["ProcessedQuestion"],
            "answer": different_answer[0],
            "different_answer": different_answer,
            "entity_linking": grounded_items,
            "sparql_query": parse["Sparql"],
            "PyQL": [x['pyql'] for x in pyql_file if x['ID'] == example['QuestionId']][0]
        })

    savejson('/home2/xhuang/PycharmProject/GLLLM/ReAct/data/WebQSP/WebQSP_test_processed', processed_examples)
