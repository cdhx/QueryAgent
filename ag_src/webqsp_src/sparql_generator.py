# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     sparql_generator
   Description :
   Author :       HX
   date：          2023/9/21
-------------------------------------------------
 
"""
__author__ = 'HX'

import re
import copy


class PyQL():
    def __init__(self):
        self.head = ''
        self.head_auto_flag = True  # If the head is automatic, it can be modified
        self.answer = ''
        self.var = []
        self.value_var = []
        self.si_value_var = []
        self.bind_mapping = {}

        self.triple_pattern = []
        self.filter = []
        self.sub_query = []
        self.aggregation = []
    def __construct_triple_pattern(self):
        return '\n' + '\n'.join(self.triple_pattern)

    def __construct_aggregation(self):
        return '\n'.join(self.aggregation)

    def __brace_indent(self, script, indent=True):
        result = ''
        if indent:
            result = script.replace('\n', '\n\t')
        return result

    @property
    def bind_var(self):
        return list(self.bind_mapping.values())

    @property
    def sparql(self):
        self.triple_pattern_text = self.__construct_triple_pattern()
        self.aggregation_text = self.__construct_aggregation()
        if self.head == '' or self.head_auto_flag == True:
            self.head = 'SELECT *'

        sparql_temp = self.head + ' {\n' + self.__brace_indent(
            self.triple_pattern_text) + '\n}\n' + self.aggregation_text
        sparql_temp = sparql_temp.replace('\n\n', '\n').replace('\n\t\n\t', '\n\t').replace('\n\t\n}', '\n}')
        return sparql_temp

    def add_triple_pattern(self, line):
        self.triple_pattern.append(line)

    def add_fact(self, s, p, o):
        # add triple pattern (s,p,o)
        s = self.__check_ent_format(s)
        p = self.__check_prop_format(p)
        o = self.__check_ent_format(o)

        self.add_triple_pattern(s + ' ' + p + ' ' + o + '.')
        self.add_triple_pattern('\n')
        self.var.extend([x for x in [s, p, o] if str(x)[0].strip() == '?'])
        self.var = list(set(self.var))

        if p == ':type.object.name':
            self.add_triple_pattern(f"FILTER(!isLiteral({o}) OR lang({o}) = '' OR langMatches(lang({o}), 'en')).")
            self.add_triple_pattern('\n')

    def add_type_constrain(self, type_id, new_var):
        # add type constrain (new_var, ns:type.object.type, type_id)
        type_id = self.__check_prop_format(type_id)
        ent = self.__check_ent_format(new_var)
        self.add_triple_pattern(ent + " :type.object.type " + type_id + ".")
        self.add_triple_pattern('\n')

        self.var.append(new_var)
        self.var = list(set(self.var))

    def add_filter(self, compare_obj1, operator, compare_obj2):
        '''
        add filter constrain
        :param compare_obj1: compare obj1
        :param operator: >,<,>=,<=,=
        :param compare_obj2: compare obj1
        '''
        operator = self.__check_op(operator)
        compare_obj1 = digit_or_var(compare_obj1)
        compare_obj2 = digit_or_var(compare_obj2)
        self.add_triple_pattern("FILTER(" + str(compare_obj1) + ' ' + operator + ' ' + str(compare_obj2) + ').')
        self.add_triple_pattern('\n')

    def add_time_constrain(self, var, relation, year):
        # add time constrain
        relation1 = relation

        if (relation.endswith('.year') or relation.endswith('.release_date')
                or relation.endswith('.initial_release_date') or relation.endswith('.discovery_date')
                or relation.endswith('.date_of_invention') or relation.endswith('.date_written')):
            relation1 = relation
        if relation.endswith('.from'):
            relation1 = relation.replace('.from', '.to')
        if relation.endswith('.start_date'):
            relation1 = relation.replace('.start_date', '.end_date')
        if relation.endswith('.from_date'):
            relation1 = relation.replace('.from_date', '.to_date')
        if relation.endswith('.to'):
            relation1 = relation
            relation = relation1.replace('.to', '.from')
        if relation.endswith('.end_date'):
            relation1 = relation
            relation = relation1.replace('.end_date', '.start_date')
        if relation.endswith('.to_date'):
            relation1 = relation
            relation = relation1.replace('.to_date', '.from_date')

        _from = relation
        _to = relation1

        from_time = f'"{year}"^^xsd:dateTime'
        to_time = f'"{year}"^^xsd:dateTime'

        self.add_triple_pattern(
            "FILTER(NOT EXISTS {{{} :{} ?sk0}} ||\nEXISTS {{{} :{} ?sk0 .\nFILTER(xsd:datetime(?sk0) >= {}) }})\n".
            format(var, _to, var, _to, to_time))

        self.add_triple_pattern(
            "FILTER(NOT EXISTS {{{} :{} ?sk1}} ||\nEXISTS {{{} :{} ?sk1 .\nFILTER(xsd:datetime(?sk1) <= {}) }})\n".
            format(var, _from, var, _from, from_time))

    def add_count(self, count_obj, new_var):
        # add count function to count_obj, the result is new_var
        count_obj = self.__check_ent_format(count_obj)

        new_var = self.__check_ent_format(new_var)
        self.__set_head('SELECT (COUNT(DISTINCT ' + count_obj + ') AS ' + new_var + ') ')

        self.var.append(new_var)
        self.var = list(set(self.var))

        return new_var

    def add_max(self, max_obj, offset=0, limit=1, data_type='string'):
        # add max fuction to max_obj, default offset=0, limit=1 (the max one)
        max_obj = '?' + max_obj if max_obj[0] != '?' else max_obj
        if data_type == 'string':
            self.aggregation.append("ORDER BY DESC(" + max_obj + ")")
        else:
            self.aggregation.append("ORDER BY DESC(xsd:float(" + max_obj + "))")
        if limit != None:
            self.aggregation.append("LIMIT " + str(limit))
        if offset != 0:
            self.aggregation.append("OFFSET " + str(offset))

    def add_min(self, min_obj, offset=0, limit=1, data_type='string'):
        # add min fuction to min_obj, default offset=0, limit=1 (the min one)
        min_obj = '?' + min_obj if min_obj[0] != '?' else min_obj

        if data_type == 'string':
            self.aggregation.append("ORDER BY (" + min_obj + ")")
        else:
            self.aggregation.append("ORDER BY (xsd:float(" + min_obj + "))")
        if limit != None:
            self.aggregation.append("LIMIT " + str(limit))
        if offset != 0:
            self.aggregation.append("OFFSET " + str(offset))

    def __set_head(self, head):
        # set the head part of the SPARQL
        self.head = head
        self.head_auto_flag = False

    def set_answer(self, answer):
        # set the answer of the SPARQL
        if '(COUNT(' not in self.head and "(AVG(" not in self.head and "(SUM(" not in self.head:
            answer = self.__check_ent_format(answer)
            self.__set_head('SELECT DISTINCT' + answer)

    def __check_op(self, op):
        # check the legality of the op
        op = op.strip()
        if op == 'gt' or op == '>':
            operator = '>'
        elif op == 'ge' or op == '>=':
            operator = '>='
        elif op == 'lt' or op == '<':
            operator = '<'
        elif op == 'le' or op == '<=':
            operator = '<='
        elif op == 'e' or op == '=':
            operator = '='
        elif op == '!=' or op == 'ne':
            operator = '!='
        else:
            raise Exception(
                'compare operator format error, op should be gt, ge, lt, le, e, ne or >, <, >=, <=, =,!= while you operator is ' + op +
                '. Besides, please carefully check if you really need to use add_filter() here if the question not need a compare constrain. ')
        return operator

    def __check_ent_format(self, entity):
        # check the legality of the entity
        entity = str(entity).strip()
        if entity == '*':
            entity = entity
        if entity[0] == '?': # variable
            entity = entity
        elif entity.startswith(':m.') or entity.startswith(':g.') :  # :mid
            entity = entity
        elif entity.startswith('m.') or entity.startswith('g.') :  # mid
            entity = ':' + entity
        elif 'http://www.w3.org/2001/XMLSchema#' in entity and '<http://www.w3.org/2001/XMLSchema#' not in entity:
            entity = '"' + entity[:entity.find('^^http')] + '"^^<' + entity[entity.find('^^http') + 2:] + '>'
            # time literal
            if '#date' in entity:
                entity = entity[:entity.find('"^^<http')] + '-08:00"^^<http://www.w3.org/2001/XMLSchema#date>'
        elif re.match('^(\-|\+)?\d+(\.\d+)?$', entity):# digit
            entity = entity
        elif '^^xsd:' in entity:# time
            entity = entity
        elif entity.endswith('"@en'): # literal
            entity = entity
        else:# other case regard as a variable without "?"
            entity = '?' + ' '.join(entity.split()).replace(' ', '_')
        return entity

    def __check_prop_format(self, prop):
        # check the legality of the property
        prop = prop.strip()
        prop_pair = prop.split('[SEP]')
        prop_pair = [x if x.startswith(':') else ':' + prop for x in prop_pair]
        return '/'.join(prop_pair)



def digit_or_var(para):

    para = str(para).strip()
    # digit in string format
    if re.match('^(\-|\+)?\d+(\.\d+)?$', para):
        if para[0] == '-':
            para = '(' + para + ')'
        if para[0] == '+':
            para = para[1:]
    # equation
    elif para.startswith('(') or para.startswith('(') or \
            para.startswith('ceil') or para.startswith('floor') or \
            para.startswith('YEAR') or para.startswith('MONTH') or \
            para.startswith('ABS') or \
            ' * ' in para or ' + ' in para or ' - ' in para or ' / ' in para:
        return para
    # time or digit
    elif '^^xsd:' in para or '^^http://www.w3.org/2001/XMLSchema' in para or '^^<http://www.w3.org/2001/XMLSchema' in para:
        return para
    # variable
    elif para[0] == '?':
        return para
    # :mid
    elif para.startswith(':m.') or para.startswith(':g.'):
        return para
    # mid
    elif para.startswith('m.') or para.startswith('g.'):
        return ':' + para
    else:
        para = '?' + ' '.join(para.split()).replace(' ', '_')
    return para
