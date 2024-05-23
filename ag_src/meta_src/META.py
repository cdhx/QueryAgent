# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name : META.py 
   Description :  
   Author :       HX
   date :    2024/1/10 10:08 
-------------------------------------------------
"""
import openai
import requests
from tqdm import tqdm
import sys
from agent_utils.config import *


from meta_src.wikienv_meta import WikiEnv
from agent_utils.ag_utils import *


def ques_ans_process(file_name):
    return_dict_list = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            return_dict = {}
            question, answers = line.split('\t')
            answer_list = answers.split('|')
            answer_list[-1] = answer_list[-1].strip()
            ent_s_idx = question.index('[')
            ent_e_idx = question.index(']')
            retrieved_ent = question[ent_s_idx+1:ent_e_idx]
            return_dict["question"] = question
            return_dict["retrieved_ent"] = retrieved_ent
            return_dict["answer"] = answer_list
            return_dict_list.append(return_dict)
    return return_dict_list

def webthink(data_item):

    config['query_cnt']=0

    env = WikiEnv()

    solve_history={'base_prompt':DATASET_PROMPT}
    question,entity_list = env.reset(data_item)
    solve_history['index'] = data_item['index']
    solve_history['question']=question
    solve_history['initial_rel']=get_rel(entity_list)
    solve_history['entity']=entity_list
    solve_history['TAO_list']=[]

    input_token_cnt=0
    output_token_cnt=0

    print('index:',data_item['index'],'\nQuestion: ',question,'\nEntity list: ',entity_list,'\nRelation for the entity: ',solve_history['initial_rel'])

    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1

        prompt=get_dynamic_history(solve_history)
        thought_action = llm(prompt,model_name=config['model'], stop=[f"\nObservation {i}:"]).replace('\n\n','\n')

        input_token_cnt += num_tokens_from_string(prompt)
        output_token_cnt += num_tokens_from_string(thought_action)

        # if the prompt is too long, give up this question
        if thought_action==PROMPT_TOO_LONG_ERROR:
            solve_history.update({'error_info':PROMPT_TOO_LONG_ERROR,'f1':0})
            return solve_history

        try:
            # the parsed thought and action
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
            print_thought('Thought ' + str(i) + ' :', thought)
            print_action('Action ' + str(i) + ' :', action)
            good_call=True
        except:
            # parser error
            print_error('Can not parse to thought and action, here is the thought and action you predicted this time:\n '+thought_action+'\n')
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

        obs,  done, info = try_step(env, action[0].lower() + action[1:])

        this_history={'step':i,'Thought':thought,'Action':action,'Observation':obs,'Refine':info.get('Refine'),'current_pyql':env.func_list, 'good_call':good_call}

        if info.get('error_in_step') is not None:
            this_history.update({'error_in_step':info['error_in_step']})
        solve_history['TAO_list'].append(this_history)

        print_obs(f"Observation {i}: {str(obs)}\n")
        print(this_history,'\n')
        if done:
            break

    # Too many steps. Forced terminate
    if not done:
        obs, done, info = try_step(env, "execute()")

    this_cost = (0.0015 * input_token_cnt + 0.002 * output_token_cnt) / 1000
    solve_history.update(info)
    solve_history.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'prompt': prompt,'input_token':input_token_cnt,'output_token':output_token_cnt,'cost':this_cost,'query_cnt':config['query_cnt']})

    del solve_history['base_prompt']
    print('===func_list===\n'+'\n'.join(solve_history['func_list'])+'\npred: '+str(solve_history['pred'][:10])+'...\ngt_answer: '+str(solve_history['gt_answer'][:10])+'...')


    return solve_history



enti_to_fact_dict = {}
with open('../../data/metaQA/kb.txt') as f:
    lines = f.readlines()
    for line in lines:
        s, r, o = line.split('|')
        if s.strip() not in enti_to_fact_dict:
            enti_to_fact_dict[s.strip()] = [line.strip()]
        else:
            enti_to_fact_dict[s.strip()].append(line.strip())
        if o.strip() not in enti_to_fact_dict:
            enti_to_fact_dict[o.strip()] = [line.strip()]
        else:
            enti_to_fact_dict[o.strip()].append(line.strip())

def get_rel(ent):
    fact=get_fact(ent)
    rel=list(set([x.split('|')[1] for x in fact]))
    return rel
def get_fact(ent):
    return enti_to_fact_dict[ent]




def metaqa_main():

    json_list = ques_ans_process(data_path)
    # random.Random(233).shuffle(json_list)

    for index,item in tqdm(enumerate(json_list) ):
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

        if data_item['index'] % 50 == 0:
            print(config)
            print('save result to: ', result_file_name)

    print(config)
    print('save result to: ', result_file_name)