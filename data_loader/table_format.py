from tabulate import tabulate
import pandas as pd
from json import loads, dumps
#TODO: whether pass data into __init__
from typing import List, Union, Optional
from pandas import DataFrame
import pandas as pd
import re
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from utils import parse_output, normalize_string_value, parse_datetime, normalize_rep_column
from functools import partial
class TableFormat:
    def __init__(self, format:str, data: Optional[Union[dict, DataFrame]] = None, use_sampling=True, save_embedding=True) -> None:
        self.format = format
        if data is not None:
            if isinstance(data, dict):
                df = normalize_rep_column(pd.DataFrame(columns=[self.normalize_col_name(c) for c in data["table"]["header"]]))
                for i, line in enumerate(data['table']['rows']):
                    df.loc[i] = line
                self.data = df
            elif isinstance(data, DataFrame):
                self.data = data
            else:
                raise ValueError("Invalid data format. Expecting a dict or DataFrame.")
            self.all_data = self.data
            # self.all_data.columns = [self.normalize_col_name(c) for c in self.all_data.columns]
            if use_sampling:
                self.data = self.data.sample(n=3, random_state=42)
            if save_embedding:
                embeddings = OpenAIEmbeddings(openai_api_base="https://api.chatanywhere.com.cn/v1", openai_api_key="sk-WZtqZEeuE0Xb6syVghDgAxdwe0ASWLkQRGxl61UI7B9RqNC4")
                row_string = []
                for i in range(len(self.all_data)):
                    row_string.append(f'# Row {i + 1} ' + ' '.join([f'{{ {self.all_data.columns[j]} : {self.all_data.iloc[i, j]} }}' for j in range(len(self.all_data.columns))]))
                    text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator='\n\n')
                    texts = text_splitter.split_text('\n\n'.join(row_string))
                    self.db = FAISS.from_texts(texts, embeddings)
    def load_data_from_dic(self, data: dict, use_sampling=True, schema_information=None):
        assert isinstance(data, dict)
        df = normalize_rep_column(pd.DataFrame(columns=[self.normalize_col_name(c) for c in data["header"]]))
        for i, line in enumerate(data['rows']):
            df.loc[i] = line
        self.data = df
        self.all_data = self.data
        if use_sampling:
            self.data = self.data.sample(n=3, random_state=42)
        if schema_information is not None:
            self.normalize_schema(schema_information)
        return self
        
        
    def format_markdown(self):
        """
        :return: the linearized text in markdown format from dict
        Markdown: 
        <Markdown grammar>\n To add a table, use three or more hyphens (---) to create each column’s header, and use pipes (|) to separate each column, every cell is separated by pipe \n"
        """
        structured_data_markdown = tabulate(
            self.data, headers=self.data.columns, tablefmt="pipe", showindex=True
        )
        return structured_data_markdown
    
    def format_nl_sep(self, table_caption:str = '', sep='|',):
        """
        Nl_sep: 
        <Grammar>\n Each table cell is separated by | , the column idx starts from 1
        """
        head = 'Col :' + sep.join(self.data.columns) + '\n'
        cells = []
        for i in range(len(self.data)):
            cells.append(f'Row {i + 1} :' + sep.join(self.data.iloc[i]))
        if table_caption:
            head = table_caption + '\n' + head
        return head + "\n".join(cells)
    
    def format_html(self, table_caption:str = ''):
        """
        if table_caption is not None, insert <caption> into the tabulate output
        """
        html = tabulate(self.data, tablefmt='unsafehtml', headers=self.data.columns, numalign="none", stralign="none", showindex='true')
        if len(table_caption):
            tag_pattern = re.compile(r'<table>')
            return tag_pattern.sub(f'<table>\n<caption>{table_caption}</caption>', html)
        return html
        
    def format_tuple(self, structured_data):
        """
        Each cell represented in tuple
        """
        
        cells = []
        for i in range(len(self.data)):
            row_string = f'Row {i + 1} :' + ' '.join([f'{self.data.columns[j]} : {self.data.iloc[i, j]}' for j in range(len(self.data.columns))])
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
                                                    'from': 'c_from',
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
        mac_dic = {'Numerical': pd.to_numeric, 'Char': normalize_string_value, 'Date': partial(pd.to_datetime, format='%Y-%m-%d')}
        for i, _ in enumerate(col_name):
            if col_name[i] in self.data.columns:
                if col_schema[i] == 'Date' or 'date' in col_name[i]:
                    try:
                        self.all_data[col_name[i]] = self.all_data[col_name[i]].apply(lambda x: parse_datetime(x))
                        self.data[col_name[i]] = self.data[col_name[i]].apply(lambda x: parse_datetime(x))
                        try:
                            self.all_data[col_name[i]] = pd.to_datetime(self.all_data[col_name[i]], format='%Y-%m-%d',errors='ignore')
                            self.data[col_name[i]] = pd.to_datetime(self.data[col_name[i]], format='%Y-%m-%d', errors='ignore')
                            self.all_data[col_name[i]] = self.all_data[col_name[i]].dt.date
                            self.data[col_name[i]] = self.data[col_name[i]].dt.date
                        except: 
                            pass
                    except:
                        print(f'Unknown Date format {self.data.head()[col_name[i]]}')
                        continue
                if col_schema[i] == 'Numerical':
                    try:
                        self.data[col_name[i]] = self.data[col_name[i]].str.replace(',', '').astype(float)
                        continue
                    except:
                        pass
                    try:
                        self.data[col_name[i]] = self.data[col_name[i]].apply(lambda x: eval(x.split('=')[1]))
                        continue
                    except:
                        pass
                    self.data[col_name[i]] = pd.to_numeric(self.data[col_name[i]], errors='ignore')
                if col_schema[i] == 'Char':
                    self.data[col_name[i]] = normalize_string_value(self.data[col_name[i]])
                        
            #TODO: whether format date in fixed format
    
    def get_all_data(self):
        self.all_data = self.data
        self.all_data.columns = [self.normalize_col_name(c) for c in self.all_data.columns]
        
    def get_sample_data(self):
            # self.data.sample(frac=0.1, replace=True, random_state=42)
            return self.data.sample(n=3, random_state=42)
        
        