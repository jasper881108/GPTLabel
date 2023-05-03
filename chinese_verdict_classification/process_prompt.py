import os
import re
import openai
import argparse
import pandas as pd
from tqdm import tqdm


system_info1 = """提取身體受傷部位,原告年齡,肇責比例資訊"""
prompt_stage1 = """幫我提取出法院同意的身體受傷部位,原告年齡,肇責比例得詳細資訊\n\n""" 
system_info1_ = """去掉沒有資訊含量的部分""" 
prompt_stage1_ = """幫我去除沒有資訊含量的語句,留下身體受傷部位,原告年齡,肇責比例\n\n"""
system_info2 = """照格式產生結構化資料，回答越精要越好"""
prompt_stage2 = """依照下文內容將身體受傷部位,原告年齡(沒有任何資訊則是為未提及原告年齡),肇責比例(如果有提及則原告+被告=100%)資訊填入至下面格式\n\n
[
{
"體傷部位":{
"頭頸部":True或False,
"臉部、耳、鼻":True或False,
"胸部":True或False,
"腹部、腰":True或False,
"骨盆":True或False,
"上肢、手":True或False,
"下肢、腳":True或False,
},
"體傷型態":{
"骨折":True或False,
"骨裂":True或False,
"擦挫傷":True或False,
"撕裂傷":True或False,
"穿刺傷":True或False,
"損傷":True或False,
"拉傷":True或False,
"灼傷":True或False,
"脱拉":True或False,
"壓迫":True或False,
"破缺損":True或False,
"壞死":True或False,
"出血":True或False,
"水腫":True或False,
"栓塞":True或False,
"剝離":True或False,
"截肢":True或False,
"失能":True或False,
"死亡":True或False,
},
 "肇事責任":{
"原告比例":0%,
"被告比例":0%,
"未提及":True或False,
},
 "原告年齡":{
"0歲-18歲":True或False,
"18歲-24歲":True或False,
"25歲-29歲":True或False,
"30歲-39歲":True或False,
"40歲-49歲":True或False,
"50歲-59歲":True或False,
"60歲-64歲":True或False,
"65歲-100歲":True或False,
"未提及":True或False,
}
]"""

default_label = dict({
"體傷部位":{
"頭頸部":False,
"臉部、耳、鼻":False,
"胸部":False,
"腹部、腰":False,
"骨盆":False,
"上肢、手":False,
"下肢、腳":False,
},
"體傷型態":{
"骨折":False,
"骨裂":False,
"擦挫傷":False,
"撕裂傷":False,
"穿刺傷":False,
"損傷":False,
"拉傷":False,
"灼傷":False,
"脱拉":False,
"壓迫":False,
"破缺損":False,
"壞死":False,
"出血":False,
"水腫":False,
"栓塞":False,
"剝離":False,
"截肢":False,
"失能":False,
"死亡":False,
},
"肇事責任":{
"原告比例":0,
"被告比例":0,
"未提及":False,
},
"原告年齡":{
"0歲-18歲":False,
"18歲-24歲":False,
"25歲-29歲":False,
"30歲-39歲":False,
"40歲-49歲":False,
"50歲-59歲":False,
"60歲-64歲":False,
"65歲-100歲":False,
"未提及":False,
}})

cost = {"gpt-3.5-turbo":0.06,"gpt-4":0.6 }

def unwrap_keys(obj, prefix='', keys=None):
    if keys is None:
        keys = []
        
    for key, value in obj.items():
        current_key = f'{prefix}{key}' if prefix else key
        keys.append(current_key)
        
        if isinstance(value, dict):
            unwrap_keys(value, prefix=f'{current_key}.', keys=keys)
    return keys

def messeage_prepare(text, prompt, system_info):
    mess = prompt + text
    message = [
        {"role": "system", "content": system_info},
        {"role": "user", "content": mess}
        ]
    return message

def main(args):     
    f = open("openai_api.txt", "r")
    api_key = f.readline()[:-1]
    openai.api_key = api_key

    model = args.model
    word_step = args.word_step
    
    f = open(args.dataset, "r")
    list_of_text = f.readlines()
    split_list_of_text = [[ text[i:i+word_step] for i in range(0, len(text), word_step)] for text in list_of_text]
    require_keys = unwrap_keys(default_label)
    
    #### Pre: Initial status, if past info exist, read and keep process
    start_idx, prompt_tokens, completion_tokens, total_tokens = 0, 0, 0, 0
    input_text, n_chunk, completions_stage1, completions_stage2 = [],[],[],[]
    
    if os.path.exists(args.path_to_file):
        df = pd.read_csv(args.path_to_file)
        start_idx = len(df)
        input_text, n_chunk = df.input_text.tolist(), df.n_chunk.tolist()
        completions_stage1, completions_stage2 = df.stage1_response.tolist(), df.response.tolist()

    epoch_bar = tqdm(range(start_idx, args.batch_size), desc="Calling OpenAI API...")
    for idx in range(start_idx, args.batch_size):

        #### Stage 1: Summarizing chunk

        completions_stage1_c = []
        split_text = split_list_of_text[idx]
        epoch_bar_chunk = tqdm(range(len(split_text)), desc="Summary chunk")
        for text in split_text:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messeage_prepare(text, prompt=prompt_stage1, system_info=system_info1),
                temperature=args.tem_for_nlg,
                )
                    
            completions_stage1_c.append(response["choices"][0]["message"]["content"])
            prompt_tokens += response["usage"]["prompt_tokens"]
            completion_tokens += response["usage"]["completion_tokens"]
            total_tokens += response["usage"]["total_tokens"]
            epoch_bar_chunk.update()
            epoch_bar_chunk.set_postfix({
                      "p_token": prompt_tokens,
                      "c_token": completion_tokens,
                      "total_token": total_tokens,
            })
        epoch_bar_chunk.close()
        completions_stage1_ = "".join(completions_stage1_c).replace("\n", "")

        if len(split_text) > 1:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messeage_prepare(completions_stage1_, prompt=prompt_stage1_, system_info=system_info1_),
                temperature=args.tem_for_nlg,
                )
                        
            completions_stage1_ = response["choices"][0]["message"]["content"].replace("\n", "")
            prompt_tokens += response["usage"]["prompt_tokens"]
            completion_tokens += response["usage"]["completion_tokens"]
            total_tokens += response["usage"]["total_tokens"]
        
        #### Stage 2: Generating specific format
        print(f"Current summary : {completions_stage1_}")
        retry_step = 0
        schema_check = True
        epoch_bar_retry = tqdm(range(args.retry_size), desc="Retry if label isn't valid") 
        while retry_step <= args.retry_size and schema_check:  
            response = openai.ChatCompletion.create(
                    model=model,
                    messages=messeage_prepare(completions_stage1_, prompt=prompt_stage2, system_info=system_info2),
                    temperature=args.tem_for_nlu + 0.1*retry_step,
                )
                
            prompt_tokens += response["usage"]["prompt_tokens"]
            completion_tokens += response["usage"]["completion_tokens"]
            total_tokens += response["usage"]["total_tokens"]
            label = response["choices"][0]["message"]["content"]

            try:
                text = re.sub("\n|%", "", label)
                st, ed = re.search("\[", text).end(), re.search("\]", text).start()
                clean_label = eval(text[st:ed])
                schema_check = require_keys != unwrap_keys(clean_label)
            except:
                pass
            
            epoch_bar_retry.update()
            retry_step += 1

        if schema_check == True:
            clean_label = "Error"
           
        epoch_bar_retry.close()

        
        input_text.append("".join(split_text))
        n_chunk.append(len(split_text))
        completions_stage1.append(completions_stage1_)
        completions_stage2.append(clean_label)
        epoch_bar.update()
        epoch_bar.set_postfix({
                         "p_token": prompt_tokens,
                         "c_token": completion_tokens,
                         "total_token": total_tokens,
                         "n_chunk":n_chunk[idx],
                         "GPT Cost NTD":total_tokens//1000*cost[args.model],
        })

        #### save file after n prompt
        if (idx+1) % args.save_step == 0:
            df = pd.DataFrame({
                "input_text":input_text,
                "n_chunk":n_chunk,
                "stage1_response":completions_stage1,
                "response":completions_stage2,
            })
            df.to_csv(args.path_to_file, header=True, index=False)

    epoch_bar.close()

    
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='30_shot_text.txt')
    parser.add_argument('--model', type=str, default='gpt-4')
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--retry_size', type=int, default=5)
    parser.add_argument('--word_step', type=int, default=5000)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--tem_for_nlg', type=float, default=0.3)
    parser.add_argument('--tem_for_nlu', type=float, default=0.0)
    parser.add_argument('--path_to_file', type=str, default='metadata/sample_json.csv')
    args = parser.parse_args()
    main(args)
