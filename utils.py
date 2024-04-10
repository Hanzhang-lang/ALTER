import re
import json
import pandas as pd
def parse_output(output: str):
    matches = re.finditer(r'([^<]*)<([^\s>]*)>', output)
    items = []
    crakets = []
    for match in matches:
        items.append(match.group(1).strip())
        crakets.append(match.group(2).strip())
    return items ,crakets

def normalize_null_value(series):
    def normalize_null(s):
        if type(s) == str and s.replace(' ', '').lower() in [
         'na', 'nan', 'none', 'null'
        ]:
            return None
        else:
            return s
    
    return series.apply(normalize_null)

def normalize_table(data, schema_information):
    col_name, col_schema = parse_output(schema_information)
    mac_dic = {'Numerical': pd.to_numeric, 'Char': normalize_null_value, 'Date': pd.to_datetime}
    for i, _ in enumerate(col_name):
        data[col_name[i]] = mac_dic[col_schema[i]](data[col_name[i]])
    return data.convert_dtypes()

def eval_fv_match(pred_list, gold_list):
        acc = 0.0
        for pred, gold in zip(pred_list, gold_list):
            if pred == gold:
                acc += 1
        acc = acc / len(pred_list)
        return acc
def eval_tabfact(output_file, gold_list, verbose=False):
    with open(output_file, 'r') as f:
        lines = f.readlines()
    specail_tokens = ['Output:', 'Output :']
    pred_label = []
    for l in lines:
        lstring = json.loads(l)['pred']
        predict_ans = ""
        for sp in specail_tokens:
            if sp in lstring:
                predict_ans = lstring.split(sp)[-1].strip()
                break
        if len(predict_ans) == 0:
            predict_ans = json.loads(l)['pred'][-3:]
            if '0' in predict_ans:
                predict_ans = '0'
            elif '1' in predict_ans:
                predict_ans = '1'
            else:
                predict_ans = '2'
        pred_label.append(predict_ans)
    if verbose:
        print(pred_label)
    return eval_fv_match(pred_label, gold_list)
    