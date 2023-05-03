import argparse
import pandas as pd
import re

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

def unwrap_keys(obj, prefix='', keys=None):
    if keys is None:
        keys = []
        
    for key, value in obj.items():
        current_key = f'{prefix}{key}' if prefix else key
        keys.append(current_key)
        
        if isinstance(value, dict):
            unwrap_keys(value, prefix=f'{current_key}.', keys=keys)
    return keys

require_keys = unwrap_keys(default_label)

def main(args):     
    df = pd.read_csv(args.json_file)

    pass_label = []
    for label in df.response.tolist():
        try:
            label = eval(label)
            if require_keys == unwrap_keys(label):
                pass_label.append(label)
            else:
                pass_label.append("Error")
        except:
            pass_label.append("Error")
    
    pass_index = [i for i in range(len(pass_label)) if pass_label[i] != "Error"]
    print(f"Pass {len(pass_index)/len(pass_label)*100:}% -> {len(pass_index)} data, other is depreciate")

    pass_label = [pass_label[i] for i in pass_index]
    pass_input_text = [df.input_text.tolist()[i] for i in pass_index]
    
    df = pd.DataFrame({
        "input_text":pass_input_text,
        "label":pass_label,
    })
    df.to_csv(args.path_to_file, header=True, index=False)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='metadata/sample_json.csv')
    parser.add_argument('--path_to_file', type=str, default='metadata/label_json.csv')
    args = parser.parse_args()
    main(args)
