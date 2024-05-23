# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name : WIKISQL.py 
   Description :  
   Author :       HX
   date :    2024/1/18 21:49 
-------------------------------------------------
"""

import openai
import requests
from agent_utils.config import *
from tqdm import tqdm

from wikisql_src.wikienv_wikisql import WikiEnv
from agent_utils.ag_utils import *



def webthink(data_item):
    config['query_cnt']=0

    env = WikiEnv()
    solve_history={'base_prompt':DATASET_PROMPT}
    question,_ = env.reset(data_item)
    print('index:',data_item['index'], '\nQuestion: ',question,'\nTable Header: ', data_item['table']['header'])
    solve_history['index'] = data_item['index']
    solve_history['header'] = data_item['table']['header']
    solve_history['question']=question
    solve_history['TAO_list']=[]

    input_token_cnt=0
    output_token_cnt=0

    n_calls, n_badcalls = 0, 0
    for i in range(1, 10):
        n_calls += 1

        prompt=get_dynamic_history(solve_history)

        # 如果触发了prompt太长的错误放弃继续reasoning
        if num_tokens_from_string(prompt)>4090:
            del solve_history['base_prompt']  # 删了这个，太长了
            this_cost = (0.0015 * input_token_cnt + 0.002 * output_token_cnt) / 1000
            solve_history.update({'error_info':PROMPT_TOO_LONG_ERROR,'f1':0,'prompt': prompt,'input_token':input_token_cnt,'output_token':output_token_cnt,'cost':this_cost,'query_cnt':config['query_cnt']})
            print('===The prompt in intermediate step is too long, stop early===')
            return solve_history

        thought_action = llm(prompt,model_name=config['model'], stop=[f"\nObservation {i}:"]).replace('\n\n','\n')


        input_token_cnt += num_tokens_from_string(prompt)
        output_token_cnt += num_tokens_from_string(thought_action)


        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
            action = action.split('\n')[0]
            print_thought('Thought ' + str(i) + ' :', thought)
            print_action('Action ' + str(i) + ' :', action)
            good_call=True
        except:
            if config['self_correction']:
                print_error('Can not parse to thought and action, here is the thought and action you predicted this time:\n ' + thought_action + '\n')
                good_call = False
                n_badcalls += 1
                n_calls += 1
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

        this_history={'step':i,'Thought':thought,'Action':action,'Observation':obs,'Refine':info.get('Refine'),'current_pyql':env.func_list,'current_sql':env.pyql.sql,'good_call':good_call}

        if info.get('error_in_step') is not None:
            this_history.update({'error_in_step':info['error_in_step']})
        solve_history['TAO_list'].append(this_history)

        print_obs(f"Observation {i}: {str(obs)}\n")
        print(this_history,'\n')
        if done:
            break

    if not done:
        obs, done, info = try_step(env, "execute()")

    this_cost = (0.0015 * input_token_cnt + 0.002 * output_token_cnt) / 1000
    solve_history.update(info)
    solve_history.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'prompt': prompt,'input_token':input_token_cnt,'output_token':output_token_cnt,'cost':this_cost,'query_cnt':config['query_cnt']})
    solve_history.update({'pred_sql':env.pyql.sql})

    del solve_history['base_prompt']
    if 'pred' not in solve_history:
        solve_history['pred']=[]
    print('===func_list===\n' + '\n'.join(solve_history['func_list']) + '\n===golden_pyql===\n' + solve_history.get(
        'golden_pyql') + '\npred: ' + str(solve_history['pred'][:10]) + (
              '...' if len(solve_history['pred']) > 10 else '') + '\ngt_answer: ' + str(
        solve_history['gt_answer'][:10]) + ('...' if len(solve_history['gt_answer']) > 10 else ''))

    return solve_history


def wikisql_main():
    json_list = readjson(data_path)
    random.Random(233).shuffle(json_list)

    for index,item in tqdm(enumerate(json_list),desc='Get linking result from cache'):
        json_list[index]['index']=index

    rs = []
    time_list=[]
    solve_history_list = []

    print("save result to: ",result_file_name)


    for data_item in json_list[:100]:

        old_time = time.time()
        solve_history = webthink(data_item)
        time_list.append(time.time() - old_time)
        rs.append(solve_history['f1'])

        solve_history['this_time'] = time_list[-1]
        solve_history['time']=sum(time_list) / len(time_list)

        total_em=len([x for x in rs if x == 1])/len(rs)
        total_f1=sum(rs) / len(rs)
        total_cost=sum([x['cost'] for x in solve_history_list+[solve_history]])/len(solve_history_list+[solve_history])
        total_query_cnt=sum([x['query_cnt'] for x in solve_history_list+[solve_history]])/len(solve_history_list+[solve_history])
        print('This f1: ',solve_history['f1'],'this input token: ', solve_history['input_token'],'this output token: ', solve_history['output_token'],'this cost: $', solve_history['cost'],'this time: ',solve_history['this_time'],'s, this query cnt: $', solve_history['query_cnt'])
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