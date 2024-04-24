import re
import json
import pandas as pd
from typing import List
from dateutil import parser
import datetime
import string
import collections
import numpy as np
def parse_output(output: str, pattern=r'([^<]*)<([^\s>]*)>'):
    """
    pattern = r'\d. (.+?): (.+)'
    pattern=r'([^<]*)<([^\s>]*)>'
    """
    matches = re.finditer(pattern, output)
    items = []
    crakets = []
    for match in matches:
        items.append(match.group(1).strip())
        crakets.append(match.group(2).strip())
    return items,crakets

def parse_specific_composition(output:str, columns:List):
    items, crakets = parse_output(output, pattern = r'\d. (.+?): (.+)')
    compositions = []
    for ind, (item, composition) in enumerate(zip(items, crakets)):
        if item in columns:
            compositions.append(item + ':' + composition)
    return compositions
    
def normalize_string_value(series, errors='coerce'):
    def normalize_null(s):
        try:
            if type(s) == str and s.replace(' ', '').lower() in [
            'na', 'nan', 'none', 'null'
            ]:
                return None
            else:
                return s.strip(' ')
        except:
            if errors == 'coerce':
                return None
    
    return series.apply(normalize_null)

def normalize_schema(data, schema_information):
    col_name, col_schema = parse_output(schema_information)
    mac_dic = {'Numerical': pd.to_numeric, 'Char': normalize_string_value, 'Date': pd.to_datetime}
    for i, _ in enumerate(col_name):
        data[col_name[i]] = mac_dic[col_schema[i]](data[col_name[i]], errors='coerce')
    return data

def parse_datetime(date_string):
    parsed_date = parser.parse(date_string)
    if parsed_date is None or not all([parsed_date.year, parsed_date.month, parsed_date.day]):
        return date_string
    normalized_date = datetime.datetime.strftime(parsed_date, "%Y-%m-%d")
    return normalized_date

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
        print(gold_list)
        print(np.where(np.array(pred_label) != np.array(gold_list)))
    return eval_fv_match(pred_label, gold_list)
    # return eval_fv_match(pred_label, gold_list)

def eval_ex_match(
        pred_list,
        gold_list,
        allow_semantic=False,
        task_name=None,
        question=None,
    ):
        def normalize_answer(s):
            def remove_articles(text):
                return re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", text)

            def whilt_space_fix(text):
                return " ".join(text.split())

            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return whilt_space_fix(remove_articles(remove_punc(lower(s))))

        def get_tokens(s):
            if not s:
                return []
            return normalize_answer(s).split()

        def compute_exact(a_gold, a_pred):
            return int(normalize_answer(a_gold) == normalize_answer(a_pred))

        def compute_f1(a_gold, a_pred):
            gold_toks = get_tokens(a_gold)
            pred_toks = get_tokens(a_pred)
            common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
            num_same = sum(common.values())
            if len(gold_toks) == 0 or len(pred_toks) == 0:
                # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
                return int(gold_toks == pred_toks)
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        if not allow_semantic:
            exact_scores = 0.0
            f1_scores = 0.0

            if task_name == "hybridqa":
                for pred, gold in zip(pred_list, gold_list):
                    exact_scores += compute_exact(gold, pred)
                    f1_scores += compute_f1(gold, pred)
            else:
                for pred, gold in zip(pred_list, gold_list):
                    exact_scores += max(compute_exact(g, pred) for g in gold)
                    f1_scores += max(compute_f1(g, pred) for g in gold)
            total = len(pred_list)
            exact_scores = exact_scores / total
            f1_scores = f1_scores / total
            return exact_scores

        else:
            assert isinstance(question, str)
            question = re.sub('\s+', ' ', question).strip().lower()
            pred_list = [str_normalize(span) for span in pred_list]
            gold_list = [str_normalize(span) for span in gold_list]
            pred_list = sorted(list(set(pred_list)))
            gold_list = sorted(list(set(gold_list)))
            # (1) 0 matches 'no', 1 matches 'yes'; 0 matches 'more', 1 matches 'less', etc.
            if len(pred_list) == 1 and len(gold_list) == 1:
                if (pred_list[0] == '0' and gold_list[0] == 'no') or (
                    pred_list[0] == '1' and gold_list[0] == 'yes'
                ):
                    return True
                question_tokens = question.split()
                try:
                    pos_or = question_tokens.index('or')
                    token_before_or, token_after_or = (
                        question_tokens[pos_or - 1],
                        question_tokens[pos_or + 1],
                    )
                    if (pred_list[0] == '0' and gold_list[0] == token_after_or) or (
                        pred_list[0] == '1' and gold_list[0] == token_before_or
                    ):
                        return True
                except Exception as e:
                    pass
            # (2) Number value (allow units) and Date substring match
            if len(pred_list) == 1 and len(gold_list) == 1:
                NUMBER_UNITS_PATTERN = re.compile(
                    '^\$*[+-]?([0-9]*[.])?[0-9]+(\s*%*|\s+\w+)$'
                )
                DATE_PATTERN = re.compile(
                    '[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})?'
                )
                DURATION_PATTERN = re.compile('(P|PT)(\d+)(Y|M|D|H|S)')
                p, g = pred_list[0], gold_list[0]
                # Restore `duration` type, e.g., from 'P3Y' -> '3'
                if re.match(DURATION_PATTERN, p):
                    p = re.match(DURATION_PATTERN, p).group(2)
                if re.match(DURATION_PATTERN, g):
                    g = re.match(DURATION_PATTERN, g).group(2)
                match = False
                num_flag, date_flag = False, False
                # Number w. unit match after string normalization.
                # Either pred or gold being number w. units suffices it.
                if re.match(NUMBER_UNITS_PATTERN, p) or re.match(
                    NUMBER_UNITS_PATTERN, g
                ):
                    num_flag = True
                # Date match after string normalization.
                # Either pred or gold being date suffices it.
                if re.match(DATE_PATTERN, p) or re.match(DATE_PATTERN, g):
                    date_flag = True
                if num_flag:
                    p_set, g_set = set(p.split()), set(g.split())
                    if p_set.issubset(g_set) or g_set.issubset(p_set):
                        match = True
                if date_flag:
                    p_set, g_set = set(p.replace('-', ' ').split()), set(
                        g.replace('-', ' ').split()
                    )
                    if p_set.issubset(g_set) or g_set.issubset(p_set):
                        match = True
                if match:
                    return True
            return check_denotation(pred_list, gold_list) 