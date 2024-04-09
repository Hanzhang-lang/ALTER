from tabulate import tabulate
import pandas as pd
from json import loads, dumps
#TODO: whether pass data into __init__
from typing import List, Union, Optional
from pandas import DataFrame
import pandas as pd
import re
class TableFormat:
    def __init__(self, format:str, data: Optional[Union[dict, DataFrame]] = None, use_sampling=True) -> None:
        self.format = format
        if data is not None:
            if isinstance(data, dict):
                df = pd.DataFrame(columns=[self.normalize_col_name(c) for c in data["table"]["header"]])
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
            
    def load_data_from_dic(self, data: dict, use_sampling=True):
        assert isinstance(data, dict)
        df = pd.DataFrame(columns=[self.normalize_col_name(c) for c in data["header"]])
        for i, line in enumerate(data['rows']):
            df.loc[i] = line
        self.data = df
        if use_sampling:
            self.data = self.data.sample(n=3, random_state=42)
        return self
        
        
    def format_markdown(self, structured_data):
        """
        :return: the linearized text in markdown format from dict
        Markdown: 
        <Markdown grammar>\n To add a table, use three or more hyphens (---) to create each column’s header, and use pipes (|) to separate each column, every cell is separated by pipe \n"
        """
        structured_data_markdown = tabulate(
            structured_data, headers=structured_data.columns, tablefmt="pipe", showindex=True
        )
        return structured_data_markdown
    
    def format_nl_sep(self, sep='|', table_caption:str = ''):
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
        html = tabulate(self.data, tablefmt='html', headers=self.data.columns, numalign="none", stralign="none", showindex='true')
        if len(table_caption):
            tag_pattern = re.compile(r'<table>')
            return tag_pattern.sub(f'<table>\n<caption>{table_caption}</caption>', html)
        return html
        
    def format_tuple(self, structured_data):
        """
        Each cell represented in tuple
        """
        pass
    
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
                                                    '#': 'num'}):
        if len(col_name) == 0:
            return 'NULL_COL'
        for c in illegal_chars:
            col_name = col_name.replace(c, illegal_chars[c])
        col_name = re.sub('_+', '_', col_name)
        return col_name 
    
    def get_all_data(self):
        self.all_data = self.data
        self.all_data.columns = [self.normalize_col_name(c) for c in self.all_data.columns]
        
    def get_sample_data(self):
            # self.data.sample(frac=0.1, replace=True, random_state=42)
            return self.data.sample(n=3, random_state=42)
        
        