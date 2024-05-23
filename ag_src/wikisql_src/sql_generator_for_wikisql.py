# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name : sql_generator_for_wikisql.py 
   Description :  
   Author :       HX
   date :    2024/1/18 21:49 
-------------------------------------------------
"""

import re
import copy
class PyQL():
    def __init__(self):
        self.answer = '*'
        self.aggregation = None
        self.table_name = None
        self.header=None
        self.condition = []
    def __construct_condition_text(self):
        if self.condition!=[]:
            return ' WHERE ' + ' AND '.join(self.condition)
        else:
            return ''

    def add_condition(self, col_name, op, condi):
        col_name=col_name.lower()
        col_name='col'+str(self.header.index(col_name))
        condi=condi.lower()
        if not (condi[-1]=="'" and condi[0]=="'"):
            condi="'" + condi + "'"
        self.condition.append(col_name+' '+op+" "+condi)


    @property
    def sql(self):
        if '-' in self.table_name:
            self.table_name = self.table_name.replace('-', '_')
        if not self.table_name.startswith('table_'):
            self.table_name = 'table_' + self.table_name

        self.condition_text = self.__construct_condition_text()
        if self.aggregation is not None:
            self.head = 'SELECT '+self.aggregation.upper()+'('+self.answer+') FROM '+self.table_name+' '
        else:
            self.head = 'SELECT ' +  self.answer + ' FROM ' + self.table_name + ' '

        sql_temp=self.head+self.condition_text
        return sql_temp


    def set_answer(self,answer,aggregation='None'):
        if aggregation!='None':
            self.aggregation=aggregation
        answer=answer.lower()
        self.answer='col'+str(self.header.index(answer))

    def add_max(self,max_var):
        max_var=max_var.lower()
        max_var='col'+str(self.header.index(max_var))
        self.set_answer(max_var)
        self.aggregation='MAX'

    def add_min(self,min_var):
        min_var=min_var.lower()
        min_var='col'+str(self.header.index(min_var))
        self.set_answer(min_var)
        self.aggregation='MIN'

    def add_count(self,count_var):
        count_var=count_var.lower()
        count_var='col'+str(self.header.index(count_var))
        self.set_answer(count_var)
        self.aggregation='COUNT'

    def add_sum(self,sum_var):
        self.header=[x.lower() for x in self.header]
        sum_var=sum_var.lower()
        sum_var='col'+str(self.header.index(sum_var))
        self.set_answer(sum_var)
        self.aggregation='SUM'

    def add_avg(self,avg_var):
        avg_var=avg_var.lower()
        avg_var='col'+str(self.header.index(avg_var))
        self.set_answer(avg_var)
        self.aggregation='AVG'
