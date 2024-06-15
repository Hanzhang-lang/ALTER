import re
import json
import pandas as pd
from pandas import DataFrame
from typing import List
from dateutil import parser
import datetime
import string
import collections
import numpy as np
import random
from datetime import timedelta
import recognizers_suite
import recognizers_suite as Recognizers
from recognizers_text import Culture, ModelResult

def str_normalize(user_input, recognition_types=None):
    """A string normalizer which recognize and normalize value based on recognizers_suite
    https://github.com/Microsoft/Recognizers-Text/tree/master/Python/"""
    user_input = str(user_input)
    user_input = user_input.replace("\\n", " ")

    def replace_by_idx_pairs(orig_str, strs_to_replace, idx_pairs):
        assert len(strs_to_replace) == len(idx_pairs)
        last_end = 0
        to_concat = []
        for idx_pair, str_to_replace in zip(idx_pairs, strs_to_replace):
            to_concat.append(orig_str[last_end : idx_pair[0]])
            to_concat.append(str_to_replace)
            last_end = idx_pair[1]
        to_concat.append(orig_str[last_end:])
        return ''.join(to_concat)

    if recognition_types is None:
        recognition_types = [
            "datetime",
            "number",
            "ordinal",
            "percentage",
            "age",
            "currency",
            "dimension",
            "temperature",
        ]
    culture = Culture.English
    for recognition_type in recognition_types:
        if re.match("\d+/\d+", user_input):
            # avoid calculating str as 1991/92
            continue
        recognized_list = getattr(
            recognizers_suite, "recognize_{}".format(recognition_type)
        )(
            user_input, culture
        )  # may match multiple parts
        strs_to_replace = []
        idx_pairs = []
        for recognized in recognized_list:
            if not recognition_type == 'datetime':
                recognized_value = recognized.resolution['value']
                if str(recognized_value).startswith("P"):
                    # if the datetime is a period:
                    continue
                else:
                    strs_to_replace.append(recognized_value)
                    idx_pairs.append((recognized.start, recognized.end + 1))
            else:
                if recognized.resolution:  # in some cases, this variable could be none.
                    if len(recognized.resolution['values']) == 1:
                        strs_to_replace.append(
                            recognized.resolution['values'][0]['timex']
                        )  # We use timex as normalization
                        idx_pairs.append((recognized.start, recognized.end + 1))

        if len(strs_to_replace) > 0:
            user_input = replace_by_idx_pairs(user_input, strs_to_replace, idx_pairs)

    if re.match("(.*)-(.*)-(.*) 00:00:00", user_input):
        user_input = user_input[: -len("00:00:00") - 1]
        # '2008-04-13 00:00:00' -> '2008-04-13'
    return user_input


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
            # compositions.append(item + ':' + composition)
            compositions.append(item + ':' + composition)
    return compositions

def parse_specific_composition_zh(output:str, columns:List):
    items, crakets = parse_output(output, pattern = r'\d. (.+?): (.+)')
    compositions = []
    for ind, (item, composition) in enumerate(zip(items, crakets)):
        if item in columns:
            # compositions.append(item + ':' + composition)
            compositions.append((item, composition))
    return compositions
    
def normalize_string_value(series, errors='coerce'):
    def normalize_null(s):
        try:
            if type(s) == str and s.replace(' ', '').lower() in [
            'na', 'nan', 'none', 'null'
            ] or len(s.strip()) == 0:
                return None
            else:
                s = re.sub(r'^"([^"]*)"$', r'\1', s.strip())
                return s.replace("–", "-").replace("—", "-").replace("―", "-").replace("−", "-").replace('\n', r'\\n').replace(u'\xa0', ' ').strip().replace('\n', ' ').strip()
        except:
            if errors == 'coerce':
                return None
    return series.apply(normalize_null)

def normalize_rep_column(data: DataFrame):
    duplicates_count = pd.Series(data.columns, index=data.columns).groupby(data.columns).cumcount()
    rename_list = [f"{col_name}_{col_count}" if col_count > 0 else col_name for (col_name, col_count) in duplicates_count.items()]
    data.columns = rename_list
    return data


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
    if parsed_date.year == datetime.datetime.now().year:
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


def normalize_number(input_str):
   # 分割字符串以获取小时、分钟和秒
    if ':' in input_str:
        hours, minutes_seconds = input_str.split(":")
        minutes, seconds = minutes_seconds.split(".")
        # 将小时、分钟和秒转换为分钟
        total_minutes = int(hours) * 60 + int(minutes) + float(seconds)
        return total_minutes
    elif ',' in input_str:
        return input_str.replace(',', '')
    elif '=' in input_str:
        return eval(input_str.split('=')[1])
    elif '%' in input_str:
        return eval(input_str.split('%')[0])
    elif input_str.replace(' ', '').lower() in ['n.a', 'n/a', 'n.a.', 'n-a', 'nan', 'none', 'null']:
        return None
    else:
        return input_str.replace('~', '').replace("–", "-").replace("—", "-").replace("―", "-").replace("−", "-").replace(' ', '').strip(' ')

def add_row_number(df: DataFrame):
    "default index start from 0"
    df.index = df.index + 1
    df = df.reset_index(names='row_number')
    return df

def composite_frame(df1, df2, reorder=False):
    n = len(df2)
    if reorder:
        insert_index = np.random.choice(range(n + 1))  
    else:
        insert_index = 0
    if insert_index == 0:  # 如果索引为0，则在开头插入
        df2 = pd.concat([df1, df2])
    elif insert_index == n:  # 如果索引为n，则在结尾插入
        df2 = pd.concat([df2, df1])
    else:  # 否则，在指定位置插入
        df2 = pd.concat([df2.iloc[:insert_index], df1, df2.iloc[insert_index:]])
    df2.reset_index(inplace=True, drop=True)
    return df2

def load_courp(p):
    courp = []
    with open(p, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip("\n").strip("\r\n")
            courp.append(line)
    return courp

def generate_text(token_count, dict):
        random_star_idx = random.randint(0, len(dict) - token_count)
        txt = dict[random_star_idx:random_star_idx + token_count]
        return ' '.join(txt)

def generate_random_text(type, dic):
    if type == 'Numerical':
        max_num = random.choice([10, 100, 1000])
        if random.random() < 0.5:
            out = '{:.2f}'.format(random.random() * max_num)
        elif random.random() < 0.7:
            out = '{:.0f}'.format(random.random() * max_num)
        else:
            # 随机保留小数点后2位
            out = str(random.random() *
                        max_num)[:len(str(max_num)) + random.randint(0, 3)]
    elif type == 'Date':
        start_date = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d')
        end_date = datetime.datetime.strptime('2010-12-31', '%Y-%m-%d')
        
        # 计算日期范围的天数差
        delta = (end_date - start_date).days
        
        # 随机选择一个天数
        random_day = random.randrange(delta)
        
        # 加上开始日期得到随机日期
        random_date = start_date + timedelta(days=random_day)
        
        # 格式化输出
        out = random_date.strftime('%Y-%m-%d')
    else:
        txt_len = random.randint(1, 3)
        out = generate_text(txt_len, dic)
        # 50% 的概率第一个字母大写
        if random.random() < 0.5:
            out = out.capitalize()
    return out

def set_random_cells_to_empty(df, k=10):
   # 获取行索引和列名的列表
   rows = df.index.tolist()
   columns = df.columns.tolist()
   
   # 确保DataFrame有足够的单元格来随机选择k个
   if len(rows) * len(columns) < k:
       raise ValueError("k is too large, there are not enough cells in the DataFrame.")
   
   # 随机生成k个不同的行索引和列名组合
   indices = np.random.choice(rows, size=k)
   cols = np.random.choice(columns, size=k)
   
   # 遍历这些组合，并将对应的单元格设置为空值''
   for i in range(k):
       df.at[indices[i], cols[i]] = ''
   
   return df


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