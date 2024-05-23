# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name : wikienv_meta.py
   Description :  
   Author :       HX
   date :    2024/1/10 10:16 
-------------------------------------------------
"""
import copy

import requests
from bs4 import BeautifulSoup

from agent_utils.ag_utils import *
# import wikipedia
from agent_utils.simlarity_search import *

# class WikiEnv(gym.Env):
class WikiEnv():
  def __init__(self):
    super().__init__()

    self.obs = ''
    self.steps = 0
    self.pred = []
    self.done = False
    self.self_correction = config['self_correction']
    self.current_result = ''
    self.current_var_value = {}
    self.relation_map = {}
    self.error_info_list = []

    self.func_list = []
    self.entity_list = {}

    self.enti_to_fact_dict = {}
    with open('../../data/metaQA/kb.txt') as f:
      lines = f.readlines()
      for line in lines:
        s, r, o = line.split('|')
        if s.strip() not in self.enti_to_fact_dict:
          self.enti_to_fact_dict[s.strip()] = [line.strip()]
        else:
          self.enti_to_fact_dict[s.strip()].append(line.strip())
        if o.strip() not in self.enti_to_fact_dict:
          self.enti_to_fact_dict[o.strip()] = [line.strip()]
        else:
          self.enti_to_fact_dict[o.strip()].append(line.strip())

  def reset(self,data_item):
    self.ent=None
    self.qid = data_item['index']
    self.question = data_item['question']
    self.entity_list = data_item['retrieved_ent']
    self.gt_answer = data_item['answer']
    self.golden_pyql = None
    self.ent = data_item['retrieved_ent']
    self.gt_answer.sort()
    return self.question, self.entity_list

  def get_reward(self):
      pred = self.pred
      f1 = f1_score(pred, self.gt_answer)
      return f1

  def get_relation(self):
    if self.func_list==[]:
      ent_list=[self.ent]
    else:
      ent_list=self.execute()

    rel=[]
    for ent in ent_list:
      fact=self.enti_to_fact_dict[ent]
      rel+=list(set([x.split('|')[1] for x in fact]))
    rel=list(set([x for x in rel if rel not in ['has_imdb_votes', 'has_imdb_rating','has_tags']]))

    if self.func_list==[]:
      rel = [x for x in rel if rel in ['written_by', 'directed_by', 'starred_actors']]

    self.obs=rel

    return rel

  def relate(self,rel):
    # rel=rel.split(',')
    rel = clean_para(rel)
    if rel not in ['release_year','in_language', 'directed_by','written_by','starred_actors','has_genre']:
      print('error here')
    self.func_list.append(rel)
    self.obs=self.execute()

  def get_ent_with_same_prop(self,rel):
    rel=clean_para(rel)
    if rel not in ['directed_by','written_by','starred_actors']:
      print('error here')
    self.func_list+=[rel,rel]
    self.obs=self.execute()
    self.obs = self.obs if len(self.obs)<10 else str(self.obs[:10])+'...'


  def get_neighbor(self,ent,rel):
    ent_fact=self.enti_to_fact_dict[ent]
    ent_fact=[x for x in ent_fact if '|'+rel+'|' in x]
    relate_ent=[]
    for fact in ent_fact:
        s,p,o=fact.split('|')
        if s==ent:
            relate_ent.append(o)
        elif o==ent:
            relate_ent.append(s)
        else:
            print('error ',fact)
    relate_ent=list(set(relate_ent))

    self.obs=relate_ent

    return relate_ent


  def execute(self):
    ent_list=copy.deepcopy([self.ent])
    for func in self.func_list:
      res=[]
      for current_ent in ent_list:
        res+=self.get_neighbor(current_ent,func)
      res=list(set(res))
      res=[x for x in res if x!=self.ent]

      ent_list=res
    return res


  def step(self, action):
    info = {"steps": self.steps, "pred": self.pred,  'em': 0, 'f1': 0,
            "qid": self.qid, "question": self.question, "gt_answer": self.gt_answer,
            'golden_pyql': self.golden_pyql}
    try:
      self.steps += 1

      action = action.strip()
      if len(self.func_list) == 2:
        if len(set(self.func_list)) == 2:
          self.func_list = [self.func_list[0]] + self.func_list
      elif len(self.func_list) == 3:
        if len(set(self.func_list)) == 1:
          self.func_list = self.func_list[:2]

      if action.startswith('execute(') and action.endswith(')') or len(self.func_list) >= 3:
        if len(self.func_list) >= 3:
          self.func_list = [x for x in self.func_list if
                          x in ['release_year', 'in_language', 'directed_by', 'written_by', 'starred_actors',
                                'has_genre']]
          self.pred = self.execute()
          self.obs = f"Step finished\n"
          self.done = True
        else:
          self.obs='This reasoning process is not finished, please carefully re-consider the action in this step.'

      elif action.startswith('get_ent_with_same_prop('):
        rel = action[len('get_ent_with_same_prop('):action.find(',')]
        self.get_ent_with_same_prop(rel)
      elif action.startswith('get_relation('):
        self.get_relation()
      elif action.startswith('relate('):
        rel = action[len('relate('):action.find(',')]
        self.relate(rel)

      else:
        self.obs = "Invalid action, next time you must choose a action from: get_relation(), relate(relation), execute()."
        raise ValueError(
          "Invalid action, next time you must choose a action from: get_relation(), relate(relation), execute().")
    except Exception as e:
      info.update({'error_in_step': str(e)})
      self.obs = 'ERROR_INFO: ' + str(e)

    info.update({'func_list': self.func_list})

    if self.done:
        f1 = self.get_reward()
        self.obs = f"Episode finished, f1 = {f1}\n"
        info.update({'em': 1 if f1 == 1 else 0, 'f1': f1,'pred':self.pred})


    return self.obs, self.done, info

