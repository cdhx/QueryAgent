# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name : wikienv_wikisql.py 
   Description :  
   Author :       HX
   date :    2024/1/18 21:49 
-------------------------------------------------
"""
import re

import requests
from bs4 import BeautifulSoup
from wikisql_src.sql_generator_for_wikisql import *
from agent_utils.ag_utils import *
# import wikipedia
from agent_utils.simlarity_search import *


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

    self.pyql = PyQL()
    self.func_list = []
    self.entity_list = {}



  def reset(self,data_item):
    self.data_item = data_item
    self.qid = self.data_item['index']
    self.question = data_item['question']
    self.gt_answer = data_item['answer']
    self.entity_list = ''
    self.golden_pyql = data_item['sql']['human_readable']
    self.pyql.header = [x.lower() for x in data_item['table']['header']]
    self.pyql.table_name = data_item['table']['id']

    self.gt_answer.sort()

    return self.question, self.entity_list
  def chain_of_check(self):
    if self.self_correction:
      pass
    self.act()
    if self.self_correction:
      pass


  def get_reward(self):
      pred = self.pred
      f1 = f1_score(pred, self.gt_answer)
      return f1
  def show_unfinised_query_result(self):
    temp_exe_res=table_result_to_list(execute_sql(self.pyql.sql))
    exe_res={}
    for k,v in temp_exe_res.items():
      if re.match('^col\d+$', k):
        exe_res[self.pyql.header[int(k[3:])]]=v
      elif 'COUNT(' in k or 'SUM(' in k or 'AVG(' in k or 'MAX(' in k or 'MIN(' in k :
        exe_res[k]=v
      else:
        raise ValueError('Column name is not one of col0 to coln  or  AGGREGATION(coln)')
    return exe_res


  def set_answer(self,answer_var,aggregation):
    answer_var = clean_para(answer_var).lower()
    aggregation = clean_para(aggregation)
    if answer_var in self.pyql.header and aggregation in ['None','MIN','MAX','COUNT','AVG','SUM']:
      self.pyql.set_answer(answer_var,aggregation)
      self.obs=self.show_unfinised_query_result()
    else:
      if config['self_correction']:
        if answer_var not in self.pyql.header and aggregation not in ['None','MIN','MAX','COUNT','AVG','SUM']:
          self.obs = "You must set_answer() again. The first parameter must be one of :"+ str(self.pyql.header)+ ', however, you set '+ answer_var+ ". The second parameter must be one of  ['NONE',,'MIN','MAX','COUNT','AVG','SUM'], however you set "+aggregation
          raise ValueError(self.obs)
        elif answer_var not in self.pyql.header:
          self.obs = "You must set_answer() again. The first parameter must be one of :"+ str(self.pyql.header)+ ', however, you set '+ answer_var+ "."
          raise ValueError(self.obs)
        elif aggregation not in ['None','MIN','MAX','COUNT','AVG','SUM']:
          self.obs = "You must set_answer() again. The second parameter must be one of  ['None',,'MIN','MAX','COUNT','AVG','SUM'], however you set "+aggregation
          raise ValueError(self.obs)
      else:
        self.pyql.set_answer(answer_var, aggregation)
        self.obs = self.show_unfinised_query_result()
  def get_column(self,column_var):
    column_var = clean_para(column_var)
    pyql_temp=copy.deepcopy(self.pyql)
    pyql_temp.set_answer(column_var)
    res = execute_sql(pyql_temp.sql)
    res = table_result_to_list(res)
    res = list(set(res[list(res.keys())[0]]))
    self.obs = res
    return res

  def add_max(self,var):
    var = clean_para(var)
    self.pyql.add_max(var)
    self.obs=self.show_unfinised_query_result()


  def add_min(self,var):
    var = clean_para(var)
    self.pyql.add_min(var)
    self.obs=self.show_unfinised_query_result()


  def add_count(self,count_var):
    count_var = clean_para(count_var)
    self.pyql.add_count(count_var)
    self.obs=self.show_unfinised_query_result()

  def add_sum(self,sum_var):
    sum_var = clean_para(sum_var)
    self.pyql.add_sum(sum_var)
    self.obs=self.show_unfinised_query_result()

  def add_avg(self,avg_var):
    avg_var = clean_para(avg_var)
    self.pyql.add_avg(avg_var)
    self.obs=self.show_unfinised_query_result()

  def add_condition(self,ob1,op,ob2):
    ob1 = clean_para(ob1)
    op = clean_para(op)
    ob2 = clean_para(ob2)
    if config['self_correction']:
      if op not in ['=', '<', '>']:
        self.obs = 'The second para of add_condition need to be one of =, <, >. However you use ' + op
        raise ValueError(self.obs)
      # use add_condition before use get_column
      if not any([x for x in self.func_list if 'get_column(' in x and ob1.strip() in x]):
        self.obs = ('I find that you use add_condition('+ob1+','+op+','+ob2+') before using get_column('+ob1+'). '
                    'I suggest you first see what is in this column by using get_column('+ob1+').  '
                    'Otherwise the value you specified('+ob2+') may not exist in this table since you donnot know what is in this column.')
        raise ValueError(self.obs)
      # add equal condition to the same column twice
      if any([x for x in self.func_list if 'add_condition(' in x and ob1.strip() in x and '=' in x]) and op =='=':
        self.obs = ('I find that you have already use add_condition() on this column:'+ob1.strip()+'. Please do not add_condition(column_name, operator, value) on the same column twice. If you restrict col1 = A and then restrict col1 = B, you will get empty result.')
        raise ValueError(self.obs)
    self.pyql.add_condition(ob1,op,ob2)
    self.obs = self.show_unfinised_query_result()

  def execute(self):
    res=table_result_to_list(execute_sql(self.pyql.sql))
    self.obs=res


  def step(self, action):
    info = {"steps": self.steps, 'em': 0, 'f1': 0,
            "qid": self.qid, "question": self.question, "gt_answer": self.gt_answer,
            'golden_pyql': self.golden_pyql}

    try:
      self.steps += 1
      action = action.strip()
      if action.startswith('get_column(') and action.endswith(')'):
        column_var = action[len('get_column('):-1]
        self.get_column(column_var)
        self.func_list.append(action)
      elif action.startswith('add_max(') and action.endswith(')'):
        max_var = action[len('add_max('):-1]
        self.add_max(max_var)
        self.func_list.append(action)
      elif action.startswith('add_min(') and action.endswith(')'):
        min_var = action[len('add_min('):-1]
        self.add_min(min_var)
        self.func_list.append(action)
      elif action.startswith('add_count(') and action.endswith(')'):
        count_var= action[len('add_count('):-1]
        self.add_count(count_var)
        self.func_list.append(action)
      elif action.startswith('add_sum(') and action.endswith(')'):
        sum_var = action[len('add_sum('):-1]
        self.add_sum(sum_var)
        self.func_list.append(action)
      elif action.startswith('add_avg(') and action.endswith(')'):
        avg_var = action[len('add_avg('):-1]
        self.add_avg(avg_var)
        self.func_list.append(action)
      elif action.startswith('add_condition(') and action.endswith(')'):
        param_list = action[len('add_condition('):-1].split(',')
        if len(param_list)==3:
          ob1, op, ob2=param_list
        elif len(param_list)>3:
          ob1 = param_list[0]
          op = param_list[1]
          ob2 = ','.join(param_list[2:])
        else:
          if config['self_correction']:
            self.obs=' add_condition(column_name, operator, value) require three parameter, however you only pass in '+str(len(param_list))+' parameters : '+str(param_list)
            raise ValueError(self.obs)
        self.add_condition(ob1, op, ob2)
        self.func_list.append(action)
      elif action.startswith('set_answer(') and action.endswith(')'):
        answer_var, aggregation = action[len('set_answer('):-1].split(',')
        self.set_answer(answer_var,aggregation)
        self.func_list.append(action)
      elif action.startswith('execute(') and action.endswith(')'):
        self.execute()
        if self.obs!={} and self.obs!=[]:
          kv_res = self.obs
          k=list(kv_res.keys())[0]
          self.pred = kv_res[k]
        else:
          self.pred=[]

        self.obs = f"Step finished\n"
        self.done = True
      else:
        if config['self_correction']:
          self.obs = "Invalid action, next time you must choose a action from: get_column(column_name), add_condition(column_name, operator, value), set_answer(column_name, aggregation_type), execute()."
          raise ValueError(self.obs)

    except Exception as e:
      info.update({'error_in_step': str(e)})
      self.obs = 'ERROR_INFO: ' + str(e)

    info.update({'func_list': self.func_list})

    if self.done:
      f1 = self.get_reward()
      self.obs = f"Episode finished, f1 = {f1}\n"
      info.update({'em': 1 if f1 == 1 else 0, 'f1': f1, 'pred': self.pred})


    return self.obs, self.done, info

