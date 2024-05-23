import requests
from bs4 import BeautifulSoup
from grail_src.sparql_generator import *
from agent_utils.ag_utils import *
# import wikipedia
from agent_utils.simlarity_search import *


class WikiEnv():
    def __init__(self):
        super().__init__()
        self.obs = ''
        self.steps = 0
        self.pred=[]
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
        # reset the basic info for this question
        self.data_item = data_item
        self.qid = self.data_item['qid']
        self.question = data_item['question']
        self.entity_list = data_item['entity_list']
        self.gt_answer = [x['answer_argument'] for x in self.data_item["answer"]]
        self.golden_pyql = [data_item['sparql_query'][data_item['sparql_query'].find('\nSELECT'):]]
        self.gt_answer.sort()

        return self.question, self.entity_list

    def get_reward(self):
        # calculate F1 score
        pred = [x.replace("http://rdf.freebase.com/ns/", "") for x in self.pred]
        f1 = f1_score(pred, self.gt_answer)
        return f1
    def chain_of_check(self, action_str):
        # reset obs
        self.obs='None'
        action_str = action_str.strip()
        action = action_str[:action_str.find('(')]
        params = [x.strip().strip('"').strip("'") for x in
                  action_str[action_str.find('(') + 1: action_str.find(')')].split(',') if x != '']
        # use self-correction, basic check
        if self.self_correction:
            self.hallucination_check(action_str)
            params = self.validity_check(action, params)
            self.rationality_check(action, params)
        # the function execute process
        success = self.act(action, params)
        # use self-correction, with execution result
        if self.self_correction:
            self.post_check(action, params, success)

    def hallucination_check(self, action_str):
        # multi action
        if action_str.count('Action') >= 1:
            self.obs = f"Invalid. You output more than one Action in current step. Think twice and output only one Action in next step. Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}."
            raise ValueError(self.obs)
        # multi thought
        if action_str.count('Thought') >= 1:
            self.obs = f"Invalid. Remember you should only output ONE Thought in each step. Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}."
            raise ValueError(self.obs)
        action = action_str[:action_str.find('(')]
        # action_list = ['get_relation', 'add_fact', 'add_max', 'add_min', 'add_count', 'add_filter', 'add_time_constrain', 'set_answer', 'execute']

        action_list = ['get_relation', 'add_fact', 'add_max', 'add_min', 'add_count', 'add_filter', 'set_answer',
                       'execute']
        if action not in action_list:
            self.obs = f"Invalid action, next time you must choose a action from: get_relation(), add_fact(), add_max(), add_min(), add_count(), add_filter(), set_answer(), execute(). Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}."
            raise ValueError(self.obs)

    def validity_check(self, action, params):
        if action == 'add_fact':
            if len(params) != 3:
                self.obs = f'add_fact(head,relation,tail) should have 3 parameters. You have {len(params)} parameters. Please check again.'
                raise ValueError(self.obs)

            # make sure head and tail in add_fact are mid not label（if is not var）
            entity_list_temp = copy.deepcopy(self.entity_list)
            for k, v in self.entity_list.items():
                entity_list_temp.update({k.lower().replace(' ', ''): v})
            if params[0].lower().replace(' ', '') in entity_list_temp.keys():
                bad_entity_id = params[0]
                entity_id = entity_list_temp[params[0].lower().replace(' ', '')]
                self.refine_info = f'Refine add_fact({",".join(params)}) to add_fact({",".join([entity_id, params[1], params[2]])})'
                print_refine(self.refine_info)
                params[0] = entity_id
                return params
            if params[2].lower().replace(' ', '') in entity_list_temp.keys():
                bad_entity_id = params[2]
                entity_id = entity_list_temp[params[2].lower().replace(' ', '')]
                self.refine_info = f'Refine add_fact({",".join(params)}) to add_fact({",".join([params[0], params[1], entity_id])})'
                print_refine(self.refine_info)
                params[2] = entity_id
                return params

            # make sure mid is in entity linking list
            if (params[0].startswith('m.') or params[0].startswith('g.')) and params[
                0] not in self.entity_list.values():
                self.obs = f'You should not use {params[0]} because it is not related to our question. Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)
            if (params[2].startswith('m.') or params[2].startswith('g.')) and params[
                2] not in self.entity_list.values():
                self.obs = f'You should not use {params[2]} because it is not related to our question. Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)
            # make sure head and tail are both mid or var
            if not (params[0].startswith('?') or params[0].startswith('m.') or params[0].startswith('g.')):
                self.obs = f'The first parameter {params[0]} is invalid in add_fact(). It must be a freebase mid or variable. Please re-think what the first parameter should be and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)


            # literal
            if not (params[2].startswith('?') or params[2].startswith('m.') or params[2].startswith(
                    'g.') or '^^http://www.w3.org/2001/XMLSchema#' in params[2]):
                self.obs = f'The third parameter {params[2]} is invalid in add_fact(). It must be a freebase mid or variable. Please re-think what the third parameter should be and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)
            # head tail are both mid
            if (params[0].startswith('m.') or params[0].startswith('g.')) and (
                    params[2].startswith('m.') or params[2].startswith('g.')):
                self.obs = f'Head and tail cannot both be mids. Use get_relation({params[0]}) or get_relation({params[2]}). Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)

        if action == 'get_relation':
            if len(params) != 1:
                self.obs = f'get_relation(para) should have 1 parameter. You have {len(params)} parameters. Please check again.'
                raise ValueError(self.obs)

            # get_relation(mention) -> get_relation(mid)
            entity_list_temp = copy.deepcopy(self.entity_list)
            for k, v in self.entity_list.items():
                entity_list_temp.update({k.lower().replace(' ', ''): v})
            if params[0].lower().replace(' ', '') in entity_list_temp.keys():
                bad_entity_id = params[0]
                entity_id = entity_list_temp[params[0].lower().replace(' ', '')]
                self.refine_info = 'Refine get_relation(' + bad_entity_id + ') to get_relation(' + entity_id + ')'
                print_refine(self.refine_info)
                params[0] = entity_id
                return params

            valid = list(self.entity_list.values()) + self.pyql.var
            if params[0] not in valid:
                cand_list = self.pyql.var
                if cand_list == []:
                    for k, v in self.entity_list.items():
                        if 'http://www.w3.org/2001/XMLSchema#' in v:
                            cand_list.append(k)
                        else:
                            cand_list.append(v)
                cand_list = sorted(list(set(cand_list)))

                cand_str = ', '.join([f'get_relation({cand})' for cand in cand_list])

                self.obs = f"You cannot query for the relations of {params[0]} because it hasn't been binded to anything. You need to first query for {cand_str}. Let's modify our approach and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}."

                raise ValueError(self.obs)



        if action == 'add_max':
            if len(params) != 1:
                self.obs = f'add_max(max_var) should have 1 parameter. You have {len(params)} parameters. Please check again.'
                raise ValueError(self.obs)
            if not params[0].startswith('?'):
                self.obs = f'The parameter in add_max must be a variable that starts with ?, but you used {params[0]}. Please check again.'
                raise ValueError(self.obs)

        if action == 'add_min':
            if len(params) != 1:
                self.obs = f'add_min(min_var) should have 1 parameter. You have {len(params)} parameters. Please check again.'
                raise ValueError(self.obs)
            if not params[0].startswith('?'):
                self.obs = f'The parameter in add_min must be a variable that starts with ?, but you used {params[0]}. Please check again.'
                raise ValueError(self.obs)

        if action == 'add_count':
            if len(params) != 2:
                self.obs = f'add_count(count_var,new_var) should have 2 parameters. You have {len(params)} parameters. Please check again.'
                raise ValueError(self.obs)
            if params[0] not in self.pyql.var:
                self.obs = f'The first parameter in add_count must be a existing variable, but you used {params[0]}. Existing variables includes: {self.pyql.var}. Please choose proper variable and set again.'
                raise ValueError(self.obs)
            if not params[1].startswith('?'):
                self.obs = f'The second parameter in add_count must be a variable that starts with ?, but you used {params[1]}. Please check again.'
                raise ValueError(self.obs)

        if action == 'set_answer':
            if len(params) != 1:
                self.obs = f'set_answer(answer_var) should have 1 parameter. You have {len(params)} parameters. Please check again.'
                raise ValueError(self.obs)
            # Existing variables includes: []
            if len(self.pyql.var) == 0:
                self.obs = f'There is no existing variable. You have to use other approach to solve the question.'
                raise ValueError(self.obs)
            # make sure answer var in an existing var
            if params[0] not in self.pyql.var:
                self.obs = f'{params[0]} is invalid. You must set a existing variable as answer. Existing variables includes: {self.pyql.var}. Please choose proper variable and set again.'
                raise ValueError(self.obs)

        if action == 'add_filter':
            if self.relation_map == {} and self.pyql.var == []:
                self.obs = f'You should use get_relation() first. Choose a parameter from {list(self.entity_list.keys())}. Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)

            if len(params) != 3:
                self.obs = f'add_filter(ob1,op,ob2) should have 3 parameters. You have {len(params)} parameters. Please check again.'
                raise ValueError(self.obs)
            if params[1] not in ['>', '<', '>=', '<=', '=', '!=']:
                self.obs = f'You used {params[1]} as operator in add_filter, which is invalid. I strongly suggest you carefully check whether a comparsion step and add_filter() is needed. If not needed and the result already meets our expectation, use set_answer() to determine which variable to return as answer. If comparision step is indeed needed, make sure the second argument is one of [>, <, >=, <=, =, !=]. Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)
            # make sure the first var in an existing var
            if params[0] not in self.pyql.var:
                self.obs = f'You choose add_filter as the action in this step. However, the first argument of add_filter should be a existing variable, while you use {params[0]}. I strongly suggest you carefully check if a comparsion step and add_filter() is needed. If not needed and the result already meets our expectation, use set_answer() to determine which variable to return as answer. If comparision step is indeed needed, make sure the first argument is one of {self.pyql.var}. Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)
            # the third parameter should not be a var
            if params[2].startswith('?'):
                self.obs = f'You choose add_filter as the action in this step. However, the third argument of add_filter should not be a variable, while you use {params[2]}. I strongly suggest you carefully check if a comparsion step is needed. If not needed and the result already meets our expectation, you can use set_answer() to determine which variable to return as answer. If comparision step is indeed needed, make sure the third argument is not a variable. Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)

        return params

    def rationality_check(self, action, params):

        if action == 'set_answer':
            # make sure the answer is not cvt
            if 'UnName_Entity' == id2label(self.current_var_value[params[0].strip('?')][0]):
                self.obs = f'You should not set {params[0]} as answer, because it has value "UnName_Entity". You can try to find the relations of {params[0]} to further constrain the query, or use other functions such as add_fact with a differernt relation. Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)

        if action == 'add_fact':
            if (params[0].startswith('?') and params[2].startswith('?')
                    and params[0] not in self.relation_map.keys() and params[2] not in self.relation_map.keys()
                    and 'http://www.w3.org/2001/XMLSchema#' not in ''.join(list(self.entity_list.values()))):
                # self.obs = f'There is something wrong in add_fact() that you output. You should consider using a mid whose relations you have already searched for, as head or tail in add_fact(). Please check again and re-generate only Thought {self.steps+1} and Action {self.steps+1}.'

                cand_mid = list(set(self.relation_map.keys()).intersection(self.entity_list.values()))
                if len(cand_mid) != 0:
                    # add_fact(?ride, amusement_parks.ride.designer, ?designer)

                    self.obs = f'You introduced two new variables at one time, which is invalid. Try replacing variable {params[0]} with {cand_mid[0]}, such as add_fact({cand_mid[0]}, {params[1]}, {params[2]}). Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'

                    for mid in cand_mid:
                        rel = params[1]
                        rels = self.relation_map[mid]
                        forward = rels['forward']
                        backward = rels['backward']
                        if rel in forward:
                            self.obs = f'You introduced two new variables at one time, which is invalid. Try replacing variable {params[0]} with {mid}, such as add_fact({mid}, {rel}, {params[2]}). Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                            break
                        elif rel in backward:
                            self.obs = f'You introduced two new variables at one time, which is invalid. Try replacing variable {params[2]} with {mid}, such as add_fact({params[0]}, {rel}, {mid}). Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                            break
                else:
                    self.obs = f'Find another approach to solve the question. Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)

            if params[1].startswith('type.type'):
                self.obs = f'Try to change an approach to solve the question. Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)


    def post_check(self, action, params, success):
        # add_fact got empty result
        if action == 'add_fact':
            if success:
                if str(self.obs).count('The value of variable ?') == 1 and 'UnName_Entity' in str(self.obs):
                    var_name = '?' + list(self.current_var_value.keys())[0]
                    self.obs = str(
                        self.obs) + f'. It seems that {var_name} is CVT node, you may use get_relation({var_name}) to further constrain the query.'
            else:
                if self.relation_map == {}:
                    cand_list = []
                    for k, v in self.entity_list.items():
                        if 'http://www.w3.org/2001/XMLSchema#' in v:
                            cand_list.append(k)
                        else:
                            cand_list.append(v)
                    cand_list = sorted(list(set(cand_list)))
                    cand_str = ', '.join([f'get_relation({cand})' for cand in cand_list])
                    self.obs = f'Got empty result. Suggestion: choose valid options from {cand_str}. Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                    raise ValueError(self.obs)

                head, relation, tail = params
                if head in self.relation_map.keys():
                    error_info = f"The relation {relation} that you choose leads to an empty result. The relations of {head} includes: {self.relation_map[head]}. Try to change a relation that matches the question best. If it still doesn't work, you may use other approach to solve the question. Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}."
                    raise ValueError(error_info)
                if tail in self.relation_map.keys():
                    error_info = f"The relation {relation} that you choose leads to an empty result. The relations of {tail} includes: {self.relation_map[tail]}. Try to change a relation that matches the question best. If it still doesn't work, you may use other approach to solve the question. Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}."
                    raise ValueError(error_info)
                error_info = f"The relation {relation} that you choose leads to an empty result. You may use other approach to solve the question. Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}."
                raise ValueError(error_info)

        # add_filter got empty result
        if action == 'add_filter' and not success:
            error_info = f'You choose add_filter as the action in this step. However, we get an empty result. I strongly suggest you carefully check if a comparsion step is needed. If not needed and the result already meets our expectation, you can use set_answer() to determine which variable to return. Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
            raise ValueError(error_info)

        # get_relation  got empty result
        if action == 'get_relation' and not success:
            cand_list = self.pyql.var
            for k, v in self.entity_list.items():
                if 'http://www.w3.org/2001/XMLSchema#' in v:
                    cand_list.append(k)
                else:
                    cand_list.append(v)
            cand_list = sorted(list(set(cand_list)))
            cand_str = ', '.join([f'get_relation({cand})' for cand in cand_list if cand != params[0]])
            error_info = f'Got empty result. Suggestion: choose from {cand_str} for next Action {self.steps + 1}. Please check again and re-generate.'

            raise ValueError(error_info)

    def act(self, action, params):
        if action == 'get_relation':
            entity = params[0]
            success = self.get_relation(entity)
            return success

        if action == 'add_fact':
            head, relation, tail = params
            raw_action = 'add_fact(' + head + ',' + relation + ',' + tail + ')'
            success, info = self.add_fact(head, relation, tail)
            if success:
                refine_action = info
                if refine_action.replace(' ', '').replace("'", '').replace('"', '') != raw_action.replace(' ',
                                                                                                          '').replace(
                    "'", '').replace('"', ''):
                    self.refine_info = 'Refine ' + raw_action + ' to ' + refine_action
                    print_refine(self.refine_info)
                self.func_list.append(refine_action)
            else:
                return False

        if action == 'add_max':
            max_var = params[0]
            self.add_max(max_var)
            self.func_list.append(f'add_max({max_var})')

        if action == 'add_min':
            min_var = params[0]
            self.add_min(min_var)
            self.func_list.append(f'add_min({min_var})')

        if action == 'add_count':
            count_var, new_var = params
            self.add_count(count_var, new_var)
            self.func_list.append(f'add_count({count_var},{new_var})')

        if action == 'add_filter':
            ob1, op, ob2 = params
            success = self.add_filter(ob1, op, ob2)
            if success:
                self.func_list.append(f'add_filter({ob1},{op},{ob2})')
            else:
                return False

        if action == 'set_answer':
            answer_var = params[0]
            self.set_answer(answer_var)
            self.func_list.append(f'set_answer({answer_var})')

        if action == 'execute':
            self.execute()
            LLM_answer_var = None
            if len(params) != 0 and params[0].strip('?') in self.current_var_value.keys():
                LLM_answer_var = params[0].strip('?')

            if self.obs != [{}] and self.obs != [] and self.obs != {}:
                kv_res = table_result_to_list(self.obs)
                k = list(kv_res.keys())[0]
                self.pred = kv_res[k]
                print('execute got result, use list(kv_res.keys())[0]: ', k)
                if len(list(kv_res.keys())) > 1 and LLM_answer_var is not None and LLM_answer_var in kv_res.keys():
                    self.pred = kv_res[LLM_answer_var]
                    print('execute got result, use LLM_answer_var: ', LLM_answer_var)
            else:
                if self.current_var_value == {}:
                    self.pred = []
                else:
                    if LLM_answer_var is None:
                        self.pred = list(self.current_var_value.values())[-1]
                        print('execute got empty result, use self.current_var_value[-1]')
                    else:
                        self.pred = self.current_var_value[LLM_answer_var]
                        print('execute got empty result, use LLM_answer_var: ', LLM_answer_var)

            self.obs = f"Step finished\n"


        return True


    def get_relation_func(self, entity_id):
        entity_id = entity_id.strip()
        entity_id = entity_id[1:] if entity_id[0] == ':' else entity_id
        entity_id = entity_id[1:] if entity_id[0] == '?' else entity_id

        if entity_id.startswith('m.'):
            sparql_head_relations = "\nPREFIX : <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  :[ENT] ?relation ?x . \n}"
            sparql_tail_relations = "\nPREFIX : <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  ?x ?relation :[ENT] . \n}"
        # literal
        elif 'http://www.w3.org/2001/XMLSchema' in entity_id and not 'http://www.w3.org/2001/XMLSchema#' not in entity_id:
            entity_id = '"' + entity_id[:entity_id.find('^^http')] + '"^^<' + entity_id[
                                                                              entity_id.find('^^http') + 2:] + '>'

            if '#date' in entity_id:
                entity_id = entity_id[:entity_id.find('"^^<http')] + '-08:00"^^<http://www.w3.org/2001/XMLSchema#date>'
            # for literal only one direction is valid(literal as the tail), the other direction use an empty sparql as a placeholder
            sparql_head_relations = "\nPREFIX : <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n :m.0yp3mmb ?relation ?x \n #[ENT] ?relation ?x. \n} limit 0"
            sparql_tail_relations = "\nPREFIX : <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  ?x ?relation [ENT] . \n}"
        else:
            entity_id = '?' + entity_id if entity_id[0] != '?' else entity_id
            if entity_id in self.pyql.var:
                sparql_head_relations = "\nPREFIX : <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  [ENT] ?relation ?x . \n" + '\n'.join(
                    self.pyql.triple_pattern) + "\n}"
                sparql_tail_relations = "\nPREFIX : <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  ?x ?relation [ENT] . \n" + '\n'.join(
                    self.pyql.triple_pattern) + "\n}"
            else:
                return [], [], False

        sparql_relations_extract_head = sparql_head_relations.replace('[ENT]', entity_id)
        head_relations = table_result_to_list(execute_sparql(sparql_relations_extract_head))

        if head_relations != {}:
            head_relations = head_relations['relation']
            head_relations = [x.replace("http://rdf.freebase.com/ns/", "") for x in head_relations if
                              'http://rdf.freebase.com/ns' in x]

        sparql_relations_extract_tail = sparql_tail_relations.replace('[ENT]', entity_id)
        tail_relations = table_result_to_list(execute_sparql(sparql_relations_extract_tail))
        if tail_relations != {}:
            tail_relations = tail_relations['relation']
            tail_relations = [x.replace("http://rdf.freebase.com/ns/", "") for x in tail_relations if
                              'http://rdf.freebase.com/ns' in x]

        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]

        head_relations = list(set(head_relations))
        tail_relations = list(set(tail_relations))

        if len(head_relations + tail_relations) > 40:
            sorted_relation = faiss_filter(self.question, list(set(head_relations + tail_relations)))[:40]
        else:
            sorted_relation = head_relations + tail_relations

        # random.shuffle(sorted_relation)

        head_relations = [x for x in sorted_relation if x in head_relations]
        tail_relations = [x for x in sorted_relation if x in tail_relations]

        head_relations.sort()
        tail_relations.sort()

        return head_relations, tail_relations, True

    def show_unfinised_query_result(self):
        # show query result of current sparql
        exe_res = execute_sparql(self.pyql.sparql)
        if exe_res != []:
            self.exe_res=exe_res
            nl_des_list = ['The value of variable ?' + k + ' is ' + str(
                [id2label(x.replace('http://rdf.freebase.com/ns/', '')) for x in v[:10]]) for k, v in
                        table_result_to_list(exe_res).items()]
            nl_des = '. '.join(nl_des_list)

            # for current_result, drop cvt
            self.current_result = '. '.join([val for val in nl_des_list if 'UnName_Entity' not in val])
            self.current_var_value = {k: [x.replace('http://rdf.freebase.com/ns/', '') for x in v] for k, v in
                                      table_result_to_list(exe_res).items()}
            return nl_des
        else:
            return []

    def get_relation(self, entity):
        if entity in self.relation_map.keys():
            head_relation = self.relation_map[entity]['forward']
            tail_relation = self.relation_map[entity]['backward']
            success = True
            # print('use cache....')
        else:
            head_relation, tail_relation, success = self.get_relation_func(entity)
        if success:
            self.obs = f'forward: {head_relation}\nbackward: {tail_relation}'
            self.relation_map[entity] = {'forward': head_relation, 'backward': tail_relation}
            print('len(head_relation+tail_relation): ', len(head_relation + tail_relation))
        else:
            self.obs = {}
        return success

    def add_fact(self, head, relation, tail):

        action_same_order = 'add_fact(' + head + ',' + relation + ',' + tail + ')'
        action_diff_order = 'add_fact(' + tail + ',' + relation + ',' + head + ')'

        # Backup for fallback
        pyql_temp = copy.deepcopy(self.pyql)
        self.pyql.add_fact(head, relation, tail)
        self.obs = self.show_unfinised_query_result()

        if self.obs != []:
            return True, action_same_order

        # based on the environmental feedback to correct the parameter, but is not necessary to invoke LLM
        if config['self_correction']:
            # change direction
            self.pyql = copy.deepcopy(pyql_temp)
            self.pyql.add_fact(tail, relation, head)
            self.obs = self.show_unfinised_query_result()

            if self.obs != []:
                return True, action_diff_order

            # try to execute only this one fact
            # if head and tail are both var, do not execute the fact alone
            if not (head.startswith('?') and tail.startswith('?')):
                # obs is still empty, consider fallback
                self.pyql = PyQL()
                self.pyql.add_fact(head, relation, tail)
                self.obs = self.show_unfinised_query_result()
                if self.obs != []:
                    return True, action_same_order

                self.pyql = PyQL()
                self.pyql.add_fact(tail, relation, head)
                self.obs = self.show_unfinised_query_result()
                if self.obs != []:
                    return True, action_diff_order

            # can not get result, drop this action, regenerate a new one
            self.pyql = copy.deepcopy(pyql_temp)

        return False, 'error'

    def set_answer(self, var):
        var = clean_para(var)
        if '(COUNT(' not in self.pyql.head:
            for mid in set(self.entity_list.values()):
                if mid.startswith('m.') or mid.startswith('g.'):
                    self.pyql.add_filter(var, '!=', mid)
        self.pyql.set_answer(var)
        self.obs = self.show_unfinised_query_result()

    def add_max(self, var):
        var = clean_para(var)
        self.pyql.add_max(var)
        self.obs = self.show_unfinised_query_result()

    def add_min(self, var):
        var = clean_para(var)
        self.pyql.add_min(var)
        self.obs = self.show_unfinised_query_result()

    def add_count(self, count_var, new_var):
        count_var = clean_para(count_var)
        new_var = clean_para(new_var)
        self.pyql.add_count(count_var, new_var)
        self.obs = self.show_unfinised_query_result()

    def add_filter(self, ob1, op, ob2):
        ob1 = clean_para(ob1)
        op = clean_para(op)
        ob2 = clean_para(ob2)


        pyql_temp = copy.deepcopy(self.pyql)

        self.pyql.add_filter(ob1, op, ob2)
        self.obs = self.show_unfinised_query_result()
        if self.obs == []:
            self.pyql = pyql_temp
            return False
        return True

    def execute(self):
        if self.exe_res!=[]:
            self.obs=self.exe_res
        else:
            self.obs = execute_sparql(self.pyql.sparql)

    def step(self, action):
        info = {"steps": self.steps,'em': 0, 'f1': 0,
                "qid": self.qid, "question": self.question, "gt_answer": self.gt_answer,
             'golden_pyql': self.golden_pyql}

        try:
            self.steps += 1

            self.refine_info = None
            action = action.strip()
            self.chain_of_check(action)
            if action[:action.find('(')] == 'execute':
                self.done = True
        except Exception as e:
            info.update({'error_in_step': str(e)})

            self.obs = 'ERROR_INFO: ' + str(e)

        info.update({'func_list': self.func_list,"potential_result": self.current_result})

        if self.refine_info is not None:
            info.update({"Refine": self.refine_info})
        if self.done:
            f1 = self.get_reward()
            self.obs = f"Episode finished, f1 = {f1}\n"
            info.update({'em': 1 if f1 == 1 else 0, 'f1': f1,'pred':self.pred})

        return self.obs, self.done, info
