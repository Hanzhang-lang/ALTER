from tabulate import tabulate
import pandas as pd
from json import loads, dumps
from typing import List, Union, Optional
from pandas import DataFrame
import pandas as pd
import re
import os
from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore, RedisStore
from utils import parse_output, normalize_string_value, parse_datetime, normalize_rep_column, normalize_number, str_normalize
from functools import partial
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TableFormat:
    def __init__(self, format: str, data: Optional[Union[dict, DataFrame]] = None, save_embedding=False, embeddings=None) -> None:
        self.format = format
        if data is not None:
            if isinstance(data, dict):
                df = normalize_rep_column(pd.DataFrame(
                    columns=[self.normalize_col_name(c) for c in data["table"]["header"]]))
                for _, line in enumerate(data['table']['rows']):
                    # rm aggregation row
                    if line[0].lower() not in ['total', 'totaal', 'totals']:
                        df.loc[len(df)] = line
                self.all_data = df
            elif isinstance(data, DataFrame):
                self.all_data = data
            else:
                raise ValueError(
                    "Invalid data format. Expecting a dict or DataFrame.")
            if save_embedding:
                self.save_row_embedding(embeddings)

    def load_data_from_dic(self, data: dict, save_embedding=True, embeddings=None, schema_information=None):
        assert isinstance(data, dict)
        df = normalize_rep_column(pd.DataFrame(
            columns=[self.normalize_col_name(c) for c in data["header"]]))
        for i, line in enumerate(data['rows']):
            df.loc[i] = line
        self.all_data = df
        if save_embedding:
            self.save_embedding(embeddings=embeddings)
        if schema_information is not None:
            self.normalize_schema(schema_information)
        return self

    def format_markdown(self, data: DataFrame, ):
        """
        :return: the linearized text in markdown format from dict
        Markdown: 
        <Markdown grammar>\n To add a table, use three or more hyphens (---) to create each column’s header, and use pipes (|) to separate each column, every cell is separated by pipe \n"
        """
        structured_data_markdown = tabulate(
            data, headers=data.columns, tablefmt="pipe", showindex=True
        )
        return structured_data_markdown

    @staticmethod
    def format_nl_sep(data: DataFrame, table_caption: str = '', sep='|',):
        """
        Nl_sep: 
        <Grammar>\n Each table cell is separated by | , the column idx starts from 1
        """
        head = 'Col :' + sep.join(data.columns) + '\n'
        cells = []
        for i in range(len(data)):
            cells.append(f'Row {i + 1} :' +
                         sep.join([str(col) for col in data.iloc[i, :]]))
        if table_caption:
            head = table_caption + '\n' + head
        return head + "\n".join(cells)

    @staticmethod
    def format_PIPE(data: DataFrame, table_caption: str = ''):
        """
        Nl_sep: 
        <Grammar>\n Each table cell is separated by | , the column idx starts from 1
        """
        linear_table = "/*\n"
        if len(table_caption):
            linear_table += "table caption : " + table_caption + "\n"

        header = "col : " + " | ".join(data.columns) + "\n"
        linear_table += header
        rows = data.values.tolist()
        for row_idx, row in enumerate(rows):
            row = [str(x) for x in row]
            line = "row : " + " | ".join(row)
            if row_idx != len(rows) - 1:
                line += "\n"
            linear_table += line
        linear_table += "\n*/\n"
        return linear_table

    @staticmethod
    def format_html(data: DataFrame, table_caption: str = ''):
        """
        if table_caption is not None, insert <caption> into the tabulate output
        """
        if len(data) == 0:
            empty_row = [None] * len(data.columns)
            data = pd.DataFrame([empty_row], columns=data.columns)
            html = tabulate(data, tablefmt='unsafehtml', headers=data.columns,
                            numalign="none", stralign="none", showindex='true',  floatfmt=".4f")
            html = re.sub(r'<tbody>.*?</tbody>', '', html, flags=re.DOTALL)
        else:
            html = tabulate(data, tablefmt='unsafehtml', headers=data.columns,
                            numalign="none", stralign="none", showindex='true', floatfmt=".4f")
        if len(table_caption):
            tag_pattern = re.compile(r'<table>')
            return tag_pattern.sub(f'<table>\n<caption>{table_caption}</caption>', html)
        return html

    def format_tuple(self, data, table_caption=''):
        """
        Each cell represented in tuple
        """

        cells = []
        for i in range(len(data)):
            row_string = f'Row {i + 1} :' + ' '.join(
                [f'{data.columns[j]} : {data.iloc[i, j]}' for j in range(len(data.columns))])
            cells.append(f'Row {i + 1} :')
        if table_caption:
            head = table_caption + '\n' + head
        return head + "\n".join(cells)

    def format_psql(self):
        return tabulate(self.all_data, tablefmt='psql', headers=self.all_data.columns, showindex='true')

    def format_json(self, structrued_data, orient="records"):
        """
        Format dataframe to json in specific orient.
        ```
        orient = 'index'
                        {
                    "row 1": {
                        "col 1": "a",
                        "col 2": "b"
                    },
                    "row 2": {
                        "col 1": "c",
                        "col 2": "d"
                    }
                }
        orient = 'records':
                    [
                {
                    "col 1": "a",
                    "col 2": "b"
                },
                {
                    "col 1": "c",
                    "col 2": "d"
                }
            ]                
        """
        structrued_data.to_json(orient=orient)
        return dumps(structrued_data)

    def normalize_col_name(self, col_name, illegal_chars={'.': '', ' ': '_',
                                                          '\\': '_',  '(': '',
                                                          ')': '', '?': '',
                                                          '\n': '_', '&': '',
                                                          ':': '_', '/': '_',
                                                          ',': '_', '-': '_',
                                                          'from': 'c_from', 'From': 'c_From', 'where': 'c_where',
                                                          '\'': '',
                                                          '%': 'percent',
                                                          '#': 'num',
                                                          # '19': 'c_19', '20': 'c_20'
                                                          }):
        if len(col_name) == 0:
            return 'NULL_COL'
        for c in illegal_chars:
            col_name = col_name.replace(c, illegal_chars[c])
        col_name = re.sub('_+', '_', col_name)
        if re.search('\d', col_name[0]):
            col_name = 'c_' + col_name
        return col_name

    def normalize_schema(self, schema_information):
        col_name, col_schema = parse_output(schema_information)
        mac_dic = {'Numerical': pd.to_numeric, 'Char': normalize_string_value,
                   'Date': partial(pd.to_datetime, format='%Y-%m-%d')}
        for i, _ in enumerate(col_name):
            if col_name[i] in self.all_data.columns:
                if col_schema[i] == 'Char':
                    self.all_data[col_name[i]] = normalize_string_value(
                        self.all_data[col_name[i]])

                if col_schema[i] == 'Date' or 'date' in col_name[i].lower():
                    try:
                        self.all_data[col_name[i]] = self.all_data[col_name[i]].apply(
                            lambda x: str_normalize(x))
                        try:
                            self.all_data[col_name[i]] = pd.to_datetime(
                                self.all_data[col_name[i]], format='%Y-%m-%d', errors='ignore')
                            self.all_data[col_name[i]
                                          ] = self.all_data[col_name[i]].dt.date
                        except:
                            pass
                    except:
                        print(
                            f'Unknown Date format {self.all_data.head()[col_name[i]]}')
                        continue
                if col_schema[i] == 'Numerical':
                    try:
                        self.all_data[col_name[i]] = self.all_data[col_name[i]].apply(
                            lambda x: normalize_number(x))
                    except:
                        pass

                    self.all_data[col_name[i]] = pd.to_numeric(
                        self.all_data[col_name[i]], errors='ignore')

        return self

    def save_row_embedding(self, embeddings, save_local=False):
        if save_local:
            store = LocalFileStore(os.path.join(dir_path, "result/.cache/"))
        else:
            store = RedisStore(redis_url="redis://localhost:6379")
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            embeddings, store, namespace="huggingface-bge"
        )
        row_string = []
        for r_index in range(len(self.all_data)):
            row_string.append(f'#Row {r_index} ' + ' '.join(
                [f'{{ {self.all_data.columns[j]} : {self.all_data.iloc[r_index, j]} }}' for j in range(len(self.all_data.columns))]))
        self.db = FAISS.from_texts(row_string, cached_embedder)

    def get_sample_data(self, sample_type: str = 'random', k: int = 3, query: str = ''):
        if k == 0:
            return self.all_data.head(0)
        if sample_type == 'random':
            try:
                return self.all_data.sample(n=k, random_state=42)
            except:
                return self.all_data
        if sample_type == 'embedding':
            retriever = self.db.as_retriever(search_kwargs={"k": k})
            pattern = re.compile(r'#Row (\d+)')
            result = retriever.invoke(query)
            row_inds = [int(pattern.search(r.page_content).group(1))
                        for r in result]
            return self.all_data.loc[row_inds]
        if sample_type == 'head':
            return self.all_data.head(k)
        if sample_type == 'all':
            return self.all_data

    def get_sample_column(self, embeddings, column_information,  threshold: float = 0.4, query: str = '', save_local=False):
        if save_local:
            store = LocalFileStore(os.path.join(dir_path, "result/.cache/"))
        else:
            store = RedisStore(redis_url="redis://localhost:6379")
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            embeddings, store, namespace="huggingface-bge"
        )
        col_names, col_infos = parse_output(
            column_information, pattern=r'([^<]*)<([^>]*)>')
        extra_col_info = []
        for i_c in range(len(col_names)):
            if col_names[i_c] in self.all_data.columns:
                extra_col_info.append(
                    f'{col_names[i_c]}: {col_infos[i_c]}' + ' '.join(self.all_data[col_names[i_c]].astype(str)))
        if len(extra_col_info) == 0:
            return []
        db = FAISS.from_texts(extra_col_info, cached_embedder)
        retriever = db.as_retriever(
            search_kwargs={"include_metadata": True, "score_threshold": threshold})
        result = retriever.invoke(query)
        return [r.page_content.split(':')[0].strip() for r in result]
