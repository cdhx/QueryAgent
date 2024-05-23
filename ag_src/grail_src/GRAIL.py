# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name : GRAILQA.py
   Description :
   Author :       HX
   date :    2023/11/24 17:48
-------------------------------------------------
"""


import openai
import requests
from tqdm import tqdm
import sys
from agent_utils.config import *



from grail_src.wikienv import WikiEnv
from agent_utils.ag_utils import *


def webthink(data_item):

    config['query_cnt']=0
    env = WikiEnv()

    question,entity_list = env.reset(data_item)
    solve_history={'base_prompt':DATASET_PROMPT}
    print('index:',data_item['index'],'  qid:',data_item['qid'],'\n',question,'\n',entity_list)
    solve_history['index'] = data_item['index']
    solve_history['question']=question
    solve_history['entity']=entity_list
    solve_history['TAO_list']=[]

    input_token_cnt=0
    output_token_cnt=0

    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1

        prompt=get_dynamic_history(solve_history)

        # if the prompt is too long, give up this question
        if num_tokens_from_string(prompt)>4090 and '16k' not in config['model'] or num_tokens_from_string(prompt)>16090:
            del solve_history['base_prompt']  # del this, too long
            this_cost = (0.0015 * input_token_cnt + 0.002 * output_token_cnt) / 1000
            solve_history.update({'error_info':PROMPT_TOO_LONG_ERROR,'f1':0,'prompt': prompt,'input_token':input_token_cnt,'output_token':output_token_cnt,'cost':this_cost,'query_cnt':config['query_cnt']})
            print('===The prompt in intermediate step is too long, stop early===')
            return solve_history
        thought_action = llm(prompt,model_name=config['model'], stop=[f"\nObservation {i}:"]).replace('\n\n','\n')


        input_token_cnt += num_tokens_from_string(prompt)
        output_token_cnt += num_tokens_from_string(thought_action)

        try:
            # the parsed thought and action
            thought, action = thought_action.strip().split(f"\nAction {i}: ")

            # if generate multi actions, only take the first one
            action = action.split('\n')[0]
            # if action.find(')') != -1:
            #     action = action[:action.find(')') + 1]

            print_thought('Thought ' + str(i) + ' :', thought)
            print_action('Action ' + str(i) + ' :', action)
            good_call=True
        except:
            if config['self_correction']:
                # parser error
                print_error('Can not parse to thought and action, here is the thought and action you predicted this time:\n ' + thought_action + '\n')
                good_call = False
                n_badcalls += 1
                n_calls += 1
                # concatenate thought to prompt, ask the LLM to generate action
                thought = thought_action.strip().split('\n')[0]
                action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:",model_name=config['model'],  stop=[f"\n"]).strip()
                print_thought(f'Thought {i} : {thought}')
                print_action(f'Action {i} (Re-generate): {action}')

                input_token_cnt += num_tokens_from_string(prompt + f"Thought {i}: {thought}\nAction {i}:")
                output_token_cnt += num_tokens_from_string(action)
            else:
                print_error(
                    'Can not parse to thought and action, here is the thought and action you predicted this time:\n ' + thought_action + '\n')
                good_call = False
                n_badcalls += 1
                thought = thought_action.strip().split('\n')[0]
                action = 'None'
                print_thought(f'Thought {i} : {thought}')
                print_action(f'Action {i} : {action}')

        obs,  done, info = try_step(env, action[0].lower() + action[1:])


        this_history={'step':i,'Thought':thought,'Action':action,'Observation':obs,'Refine':info.get('Refine'),'current_pyql':env.func_list,'current_sparql':env.pyql.sparql,'good_call':good_call}

        if info.get('error_in_step') is not None:
            this_history.update({'error_in_step':info['error_in_step']})
        solve_history['TAO_list'].append(this_history)

        print_obs(f"Observation {i}: {str(obs)}\n")
        print(this_history,'\n')
        if done:
            break


    # Too many steps. Forced terminate
    if not done and config['self_correction'] :
        # let llm choose one variable as the answer
        if info['potential_result'] != '':
            prompt = f"I will give you a question and some results. Your only job is to choose a variable from the results that best matches the question and output set_answer(?variable_name). Note that all variables start with a question mark '?'. \nHere is an example:\nQuestion : what taylor swift album called \nResult : The value of variable ?date is ['2012-08', '2012-07', '2012-12', '2009-01']. The value of variable ?album is [' Red', ' Speak Now', ' Fearless']\nOutput : set_answer(?album)\nQuestion : {question}\nResult : {info['potential_result']}\nOutput : "
            action = llm(prompt, model_name=config['model'],  stop=[f"\n"]).strip()
            input_token_cnt += num_tokens_from_string(prompt)
            output_token_cnt += num_tokens_from_string(action)
            var = re.findall(r"set_answer\((\?\w+)\)", action)
            if len(var) == 0:
                obs, done, info = try_step(env, "execute()")
            else:
                obs, done, info = try_step(env, f"execute({var[0]})")
        else:
            obs, done, info = try_step(env, "execute()")

    this_cost = (0.0015 * input_token_cnt + 0.002 * output_token_cnt) / 1000
    solve_history.update(info)
    solve_history.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'prompt': prompt,'input_token':input_token_cnt,'output_token':output_token_cnt,'cost':this_cost,'query_cnt':config['query_cnt']})

    solve_history.update({'qid':info['qid'],'pred':info.get('pred') if info.get('pred') is not None else [],'gt_answer':info['gt_answer'],'f1':info.get('f1') if info.get('f1') is not None else 0,'func_list':env.func_list,'golden_pyql':info.get('golden_pyql'),'n_calls': n_calls, 'n_badcalls': n_badcalls, 'prompt': prompt})
    del solve_history['base_prompt'] # del this, too long
    print('===func_list===\n' + '\n'.join(solve_history['func_list']) + '\n===golden_pyql===\n' + str(
        '\n'.join(solve_history.get('golden_pyql'))) + '\npred: ' + str(solve_history['pred'][:10]) + (
              '...' if len(solve_history['pred']) > 10 else '') + '\ngt_answer: ' + str(
        solve_history['gt_answer'][:10]) + ('...' if len(solve_history['gt_answer']) > 10 else ''))

    return solve_history


import random


def grailqa_main():
    json_list = readjson(data_path)
    random.Random(233).shuffle(json_list)

    if config['golden_el'] == False:
        el_result=readjson(linking_path)
        el_result={k:v for k,v in el_result.items() if 'test' not in k}


    for index,item in tqdm(enumerate(json_list),desc='Get linking result from cache'):
        if config['golden_el']==True:
            json_list[index]['entity_list'] = {x['friendly_name'].strip():x['id'] for x in item['graph_query']['nodes'] if x['node_type'] == 'entity'}

        else:
            json_list[index]['entity_list'] = {k.strip():v[0] for k,v in [v for k,v in el_result.items() if k==str(item['qid'])][0].items()}
        json_list[index]['index']=index

    print('total number of questions:',len(json_list))

    rs = []
    time_list=[]
    solve_history_list = []

    print("save result to: ",result_file_name)


    for data_item in json_list[:100]:


        old_time = time.time()
        solve_history = webthink(data_item)

        time_list.append(time.time() - old_time)
        rs.append(solve_history['f1'])

        #time
        solve_history['this_time'] = time_list[-1]
        solve_history['time']=sum(time_list) / len(time_list)

        total_em=len([x for x in rs if x == 1])/len(rs)
        total_f1=sum(rs) / len(rs)
        total_cost=sum([x['cost'] for x in solve_history_list+[solve_history]])/len(solve_history_list+[solve_history])
        total_query_cnt=sum([x['query_cnt'] for x in solve_history_list+[solve_history]])/len(solve_history_list+[solve_history])
        print('This f1: ',solve_history['f1'],'this input token: ', solve_history['input_token'],'this output token: ', solve_history['output_token'],'this cost: $', solve_history['cost'],'this time: ',solve_history['this_time'],'s, this query cnt: ', solve_history['query_cnt'])
        print('count:', len(rs), ', f1: ', total_f1, ', em: ', total_em, ', time: ',solve_history['time'], 's, cost: $',total_cost,', query_cnt: ',total_query_cnt)
        print('-----------\n')
        solve_history.update({'total_f1':total_f1,'total_em': total_em})
        solve_history_list.append(solve_history)
        savejson(result_file_name, solve_history_list)

        if data_item['index']%50==0:
            print(config)
            print('save result to: ',result_file_name)


    print(config)
    print('save result to: ',result_file_name)