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
            to_concat.append(orig_str[last_end: idx_pair[0]])
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
                        idx_pairs.append(
                            (recognized.start, recognized.end + 1))

        if len(strs_to_replace) > 0:
            user_input = replace_by_idx_pairs(
                user_input, strs_to_replace, idx_pairs)

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
    return items, crakets


def parse_specific_composition(output: str, columns: List):
    items, crakets = parse_output(output, pattern=r'\d. (.+?): (.+)')
    compositions = []
    for ind, (item, composition) in enumerate(zip(items, crakets)):
        if item in columns:
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
    duplicates_count = pd.Series(
        data.columns, index=data.columns).groupby(data.columns).cumcount()
    rename_list = [f"{col_name}_{col_count}" if col_count > 0 else col_name for (
        col_name, col_count) in duplicates_count.items()]
    data.columns = rename_list
    return data


def normalize_schema(data, schema_information):
    col_name, col_schema = parse_output(schema_information)
    mac_dic = {'Numerical': pd.to_numeric,
               'Char': normalize_string_value, 'Date': pd.to_datetime}
    for i, _ in enumerate(col_name):
        data[col_name[i]] = mac_dic[col_schema[i]](
            data[col_name[i]], errors='coerce')
    return data


def parse_datetime(date_string):
    parsed_date = parser.parse(date_string)
    if parsed_date is None or not all([parsed_date.year, parsed_date.month, parsed_date.day]):
        return date_string
    if parsed_date.year == datetime.datetime.now().year:
        return date_string
    normalized_date = datetime.datetime.strftime(parsed_date, "%Y-%m-%d")
    return normalized_date


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
            out = str(random.random() *
                      max_num)[:len(str(max_num)) + random.randint(0, 3)]
    elif type == 'Date':
        start_date = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d')
        end_date = datetime.datetime.strptime('2010-12-31', '%Y-%m-%d')
        delta = (end_date - start_date).days
        random_day = random.randrange(delta)
        random_date = start_date + timedelta(days=random_day)
        out = random_date.strftime('%Y-%m-%d')
    else:
        txt_len = random.randint(1, 3)
        out = generate_text(txt_len, dic)
        if random.random() < 0.5:
            out = out.capitalize()
    return out


def set_random_cells_to_empty(df, k=10):
    rows = df.index.tolist()
    columns = df.columns.tolist()
    if len(rows) * len(columns) < k:
        raise ValueError(
            "k is too large, there are not enough cells in the DataFrame.")
    indices = np.random.choice(rows, size=k)
    cols = np.random.choice(columns, size=k)
    for i in range(k):
        df.at[indices[i], cols[i]] = ''
    return df
