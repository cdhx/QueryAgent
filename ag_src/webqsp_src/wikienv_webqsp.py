import requests
from bs4 import BeautifulSoup
from webqsp_src.sparql_generator import *
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

        self.pyql_manager = []

    def reset(self,data_item):
        # reset the basic info for this question
        self.data_item = data_item
        self.qid = self.data_item['qid']
        self.question = data_item['question']
        self.entity_list = data_item['entity_list']
        self.gt_answer = data_item['answer']
        self.different_gt_answer = data_item['different_answer']
        self.golden_pyql = [line for line in data_item.get('PyQL').split('\n') if
                            'PyQL()' not in line and '!=' not in line]

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
            self.obs = f"Invalid. You output too many actions in current step. Remember you should only take ONE Action in each step. Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}."
            raise ValueError(self.obs)
        # multi thought
        if action_str.count('Thought') >= 1:
            self.obs = f"Invalid. Remember you should only output ONE Thought in each step. Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}."
            raise ValueError(self.obs)
        action = action_str[:action_str.find('(')]
        # action_list = ['get_relation', 'add_fact', 'add_max', 'add_min', 'add_count', 'add_filter', 'add_time_constrain', 'set_answer', 'execute']

        action_list = ['get_relation', 'add_fact', 'add_max', 'add_min', 'add_time_constrain', 'set_answer', 'execute']
        if action not in action_list:
            self.obs = f"Invalid action, next time you must choose a action from: get_relation(), add_fact(), add_max(), add_min(), add_time_constrain(), set_answer(), execute(). Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}."
            raise ValueError(self.obs)

    def validity_check(self, action, params):
        if action == 'add_fact':

            if len(params) != 3:
                self.obs = f'add_fact(head,relation,tail) should have 3 parameters. You have {len(params)} parameters. Please check again.'
                raise ValueError(self.obs)

            # make sure mid is in entity linking list
            if params[0].startswith('m.') and params[0] not in self.entity_list.values():
                self.obs = f'You should not use {params[0]} because it is not related to our question. Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)
            if params[2].startswith('m.') and params[2] not in self.entity_list.values():
                self.obs = f'You should not use {params[2]} because it is not related to our question. Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)


            # make sure head and tail in add_fact are mid not label（if is not var）
            entity_list_temp = {}
            for k, v in self.entity_list.items():
                if str(v).startswith('m.'):
                    entity_list_temp.update({k.lower().replace(' ', ''): v})
            if params[2].lower().replace(' ', '') in entity_list_temp.keys():
                entity_id = entity_list_temp[params[2].lower().replace(' ', '')]
                self.refine_info = f'Refine add_fact({",".join(params)}) to add_fact({",".join([params[0], params[1], entity_id])})'
                print_refine(self.refine_info)
                params[2] = entity_id


            # make sure head and tail are both mid or var
            if not (params[0].startswith('?') or params[0].startswith('m.')):
                self.obs = f'The first parameter {params[0]} is invalid in add_fact. It must be a freebase mid or variable. Please re-think what the first parameter should be and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)

            if params[1] in ['!=', '=', '<', '<=', '>', '>=']:
                self.obs = f'There is no need to set constrain on {params[0]} and {params[2]}. We will automatically filter out inappropriate results for you. Use set_answer if the answer is complete enough. Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)

            is_numeric = False
            try:
                year = int(params[2])
                is_numeric = True
            except Exception:
                pass
            # if the third parameter is digit, it is likely the llm mistakenly use add_fact raather than add_time_constraint
            if is_numeric:
                self.obs = f'The third parameter {params[2]} is invalid in add_fact. I noticed it may be year. Please check whether you want to use add_time_constrain({params[0]}, {params[1]}, {int(params[2])}) and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)

            if not (params[2].startswith('?') or params[2].startswith('m.')):
                self.obs = f'The third parameter {params[2]} is invalid in add_fact. It must be a freebase mid or variable. Please re-think what the third parameter should be and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)

            # 1st para and 3st para is the same
            if params[0] == params[2]:
                self.obs = f'The third parameter should not be the same as the first parameter. Consider naming a new variable. Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)

        if action == 'get_relation':
            if len(params) != 1:
                self.obs = f'get_relation(para) should have 1 parameter. You have {len(params)} parameters. Please check again.'
                raise ValueError(self.obs)
            #   if not (params[0].startswith('?') or params[0].startswith('m.')):
            if not (params[0] in self.pyql.var or params[0] in self.entity_list.values()):
                cand_mids = [val for val in self.entity_list.values() if str(val).startswith('m.')]
                # get_relation(?senator)
                if self.pyql.var == []:
                    cand_str = ', '.join([f'get_relation({mid})' for mid in cand_mids])
                    self.obs = f'{params[0]} is invalid in get_relation. Consider using a mid from entity list, such as {cand_str}. Please check again.'
                    raise ValueError(self.obs)
                else:
                    self.obs = f'{params[0]} is invalid in get_relation. It must be a existing variable, such as {self.pyql.var} or a mid from entity list, such as {cand_mids}. Please check again.'
                    raise ValueError(self.obs)

        if action == 'add_max':
            if len(params) != 1:
                self.obs = f'add_max(max_var) is used to constrain the query by only returning the result when max_var is the biggest. Only one parameter is needed. You should choose max_var from {self.pyql.var}. Please check again.'
                raise ValueError(self.obs)
            if params[0] not in self.pyql.var:
                self.obs = f'add_max(max_var) is used to constrain the query by only returning the result when max_var is the biggest. You used an unexisting variable {params[0]}. You should choose max_var from {self.pyql.var}. Please check again.'
                raise ValueError(self.obs)

        if action == 'add_min':
            if len(params) != 1:
                self.obs = f'add_min(min_var) is used to constrain the query by only returning the result when min_var is the smallest. Only one parameter is needed. You should choose min_var from {self.pyql.var}. Please check again.'
                raise ValueError(self.obs)
            if params[0] not in self.pyql.var:
                self.obs = f'add_min(min_var) is used to constrain the query by only returning the result when min_var is the smallest. You used an unexisting variable {params[0]}. You should choose min_var from {self.pyql.var}. Please check again.'
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

        if action == 'add_time_constrain':
            if len(params) != 3:
                self.obs = f'add_time_constrain(var, relation, year) should have 3 parameters. You have {len(params)} parameters. Please check again.'
                raise ValueError(self.obs)
            if params[0] not in self.pyql.var:
                self.obs = f'{params[0]} is invalid. You must set a existing variable as first parameter. Existing variables includes: {self.pyql.var}. Please choose proper variable and set again.'
                raise ValueError(self.obs)
            try:
                year = params[2]
                year = year[:4]
                year = int(year)
            except Exception:
                self.obs = f'The third parameter in add_time_constrain should be a specific year, but you used {params[2]}. An example: add_time_constrain(?team, sports.sports_team_roster.from, 2010). Please check again.'
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
        if action == 'add_time_constrain':
            # If the first argument has not queried for relations
            # it is likely that the relation in add_time_constrain is wrong.
            if params[0] not in self.relation_map.keys():
                cand = [var for var in self.pyql.var if var != params[0]]
                self.obs = f'You used {params[0]} as first parameter, but it may not be correct. Please choose correct variable from {cand} and re-generate add_time_constrain.'
                raise ValueError(self.obs)
            if params[1] not in self.relation_map[params[0]]:
                self.obs = f"The relation {params[1]} that you chose may not be correct. Try to choose relation that matches the question best from {self.relation_map[params[0]]}. Please re-think and re-generate add_time_constrain with a proper relation."
                raise ValueError(self.obs)
            if len(self.func_list) == 0:
                self.obs = f'You should use get_relation() first. Please re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)

        if action == 'set_answer':
            # make sure the answer is not cvt
            if 'UnName_Entity' == id2label(self.current_var_value[params[0].strip('?')][0]):
                self.obs = f'You should not set {params[0]} as answer, because it has value "UnName_Entity". You can try to find the relations of {params[0]} to further constrain the query, or use other functions such as add_fact with a differernt relation. Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)

    def post_check(self, action, params, success):

        if action == 'add_fact':
            if success:
                return
                pattern = r"\?(\w+)\s+is\s+(\[.*?\])"
                # matches  =  [('team', "['UnName_Entity', 'UnName_Entity', 'UnName_Entity']"), ('team_name', "['Newcastle Jets FC', 'Wigan Athletic F.C.', 'Aston Villa F.C.']")]
                matches = re.findall(pattern, str(self.obs))
                if len(matches) == 1 and 'UnName_Entity' in str(self.obs):
                    var_name = '?' + matches[0][0]
                    self.obs = str(
                        self.obs) + f'. It seems that {var_name} is CVT node because it contains \'UnName_Entity\', you may use get_relation({var_name}) to further constrain the query.'

                return

            if self.relation_map == {}:
                cand_list = []
                for k, v in self.entity_list.items():
                    cand_list.append(v)
                cand_list = sorted(list(set(cand_list)))
                cand_str = ', '.join([f'get_relation({cand})' for cand in cand_list])
                self.obs = f'The result is empty. You should choose valid options from {cand_str}. Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}.'
                raise ValueError(self.obs)

            head, relation, tail = params
            if head in self.relation_map.keys():
                error_info = f"The relation {relation} that you choose leads to an empty result. Re-think the question again and choose another relation from {self.relation_map[head]}. Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}."
                raise ValueError(error_info)

            if tail in self.relation_map.keys():
                error_info = f"The relation {relation} that you choose leads to an empty result. Re-think the question again and choose another relation from {self.relation_map[tail]}. Please check again and re-generate only Thought {self.steps + 1} and Action {self.steps + 1}."
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
                self.pyql_manager.append(copy.deepcopy(self.pyql))
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

        if action == 'add_time_constrain':
            var1, var2, var3 = params
            self.add_time_constrain(var1, var2, var3)
            self.func_list.append(f'add_time_constrain({var1},{var2},{var3})')

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

        original_entity_id = entity_id

        entity_id = entity_id.strip()
        entity_id = entity_id[1:] if entity_id[0] == ':' else entity_id
        entity_id = entity_id[1:] if entity_id[0] == '?' else entity_id

        entity_list_temp = copy.deepcopy(self.entity_list)
        for k, v in entity_list_temp.items():
            self.entity_list.update({k.lower(): v})

        if original_entity_id.lower() in self.entity_list.keys():
            bad_entity_id = original_entity_id
            entity_id = self.entity_list[original_entity_id.lower()]
            self.refine_info = 'Refine get_relation(' + bad_entity_id + ') to get_relation(' + entity_id + ')'
            print_refine(self.refine_info)
        # entity
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
                self.obs = "The parameter of get_relation() must be a freebase mid or a existing variable as answer. Existing variables invludes: ", self.pyql.var, '. However, you set ', entity_id, ', which is an unexisting variable.'
                return [], [], False


        sparql_relations_extract_head = sparql_head_relations.replace('[ENT]', entity_id)
        head_relations = table_result_to_list(execute_sparql(sparql_relations_extract_head))

        if head_relations != []:
            head_relations = head_relations['relation']

            head_relations = [x.replace("http://rdf.freebase.com/ns/", "") for x in head_relations if
                              'http://rdf.freebase.com/ns' in x]


        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]

        head_relations = list(set(head_relations))

        if len(head_relations) > 40:
            sorted_relation = faiss_filter(self.question, head_relations)[:40]
        else:
            sorted_relation = head_relations

        head_relations = [x for x in sorted_relation if x in head_relations]

        head_relations = list(set(head_relations))

        head_relations.sort()

        tail_relations = []
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
        entity = clean_para(entity)

        head_relation, tail_relation, success = self.get_relation_func(entity)
        if success:
            self.obs = head_relation
            self.relation_map[entity] = head_relation
            print('len(head_relation+tail_relation): ', len(head_relation + tail_relation))
        else:
            self.obs = []
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
        if var in self.pyql.var:
            for mid in set(self.entity_list.values()):
                self.pyql.add_filter(var, '!=', mid)
            self.pyql.set_answer(var)
            self.obs = self.show_unfinised_query_result()

    def add_max(self, var):
        var = clean_para(var)

        if self.current_var_value != {}:
            is_numeric = False
            try:
                val = float(self.current_var_value[var.strip('?')][0])
                is_numeric = True
            except ValueError:
                pass
            if is_numeric:
                self.pyql.add_max(var, data_type='float')
            elif '-08:00' in self.current_var_value[var.strip('?')][0]:
                self.pyql.add_max(var, data_type='string')
        else:
            self.pyql.add_max(var)

        self.obs = self.show_unfinised_query_result()

    def add_min(self, var):
        var = clean_para(var)

        if self.current_var_value != {}:
            is_numeric = False
            try:
                val = float(self.current_var_value[var.strip('?')][0])
                is_numeric = True
            except ValueError:
                pass
            if is_numeric:
                self.pyql.add_min(var, data_type='float')
            elif '-08:00' in self.current_var_value[var.strip('?')][0]:
                self.pyql.add_min(var, data_type='string')
        else:
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

    def add_time_constrain(self, var1, var2, var3):
        var1 = clean_para(var1)
        var2 = clean_para(var2)
        var3 = clean_para(var3)

        self.pyql.add_time_constrain(var1, var2, var3)
        self.obs = self.show_unfinised_query_result()
        temp = copy.deepcopy(self.pyql)
        if self.obs == [] and len(self.pyql_manager) > 1:
            for pyql in reversed(self.pyql_manager[:-1]):
                self.pyql = copy.deepcopy(pyql)
                self.pyql.add_time_constrain(var1, var2, var3)
                self.obs = self.show_unfinised_query_result()
                if self.obs != []:
                    # print_action("Fallback successful")
                    break

        if self.obs == []:
            self.pyql = temp

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
