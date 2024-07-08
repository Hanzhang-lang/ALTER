from datasets import load_dataset
import pandas as pd
import re
import os

def normalize_col_name(col_name, illegal_chars={'.': '', ' ': '_',
                                                '\\': '_',  '(': '',
                                                ')': '', '?': '',
                                                '\n': '_', '&': '',
                                                ':': '_', '/': '_',
                                                ',': '_', '-': '_',
                                                'from': 'c_from',
                                                '\'': '',
                                                '%': 'percent',
                                                '#': 'num',
                                                '19': 'c_19', '20': 'c_20'}):
    if len(col_name) == 0:
        return 'NULL_COL'
    for c in illegal_chars:
        col_name = col_name.replace(c, illegal_chars[c])
    col_name = re.sub('_+', '_', col_name)
    if re.search('\d', col_name[0]):
        col_name = 'c_' + col_name
    return col_name


class TableLoader:
    def __init__(self, table_name: str, split: str = None, use_sample: bool = True, small_test: bool = False, cache_dir=None) -> None:
        """

        """
        self.table_name = table_name
        self.dataset = self.load_table(use_sample, split=split, cache_dir=cache_dir)
        if small_test:
            self.dataset = self.dataset.filter(
                lambda example: example['small_test'])

    def load_table(self, use_sample: bool = True, split: str = None, cache_dir = None ):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        if self.table_name == 'fetaqa':
            dataset = load_dataset('DongfuJiang/FeTaQA',  cache_dir=cache_dir)
        else:
            dataset = load_dataset(
            os.path.join(dir_path, f"datasets/{self.table_name}.py"), verification_mode="no_checks", cache_dir=cache_dir)
        if split:
            dataset = dataset[split]
        if use_sample and len(dataset) > 300:
            shuffled_dataset = dataset.shuffle(seed=42)
            return shuffled_dataset.select(range(300))
        return dataset

    def normalize_table(self, _line: dict):
        if self.table_name == 'fetaqa':
            return {
                "id": _line['feta_id'],
                "title": "",
                "context": "",
                "table": {
                    "header": _line['table_array'][0],
                    "rows": _line['table_array'][1:],
                    "caption": _line['table_page_title'],
                    "id": _line['table_source_json'],
                },
                "query": _line['question'],
                "label": _line['answer']}
        
        if self.table_name == 'tabfact':
            if str(_line["label"]) == "0":
                label = "0"
            elif str(_line["label"]) == "1":
                label = "1"
            else:
                label = "2"
            return {
                "id": _line['id'],
                "title": "",
                "context": "",
                "table": {
                    "id": _line['table']['id'],
                    "header": _line['table']['header'],
                    "rows": _line['table']['rows'],
                    "caption": _line['table']['caption'],
                },
                "query": _line["statement"],
                "label": label,
            }
        if self.table_name == 'wikitable':
            return {
                "id": _line['id'],
                "title": "",
                "context": "",
                "table": {
                    "header": _line['table']['header'],
                    "rows": _line['table']['rows'],
                    "caption": _line['caption'],
                    "id": _line['table']['name'],
                },
                "query": _line['question'],
                "label": _line['answers']}
            
        if self.table_name == 'totto':
            return {
                "title": _line['table_page_title'],
                "context": "",
                "table": {
                    "header": _line['table_rows'][0],
                    "rows": _line['table_rows'][1:],
                    "caption": _line['table_section_title'],
                    "header_hierarchy": _line['table_header_hierarchy'],
                },
                "query": f"Produce a one-sentence description for each highlighted cells ({str(_line['highlighted_cells'])}) of the table.",
                "label": _line["final_sentences"],
            }
        if self.table_name == 'sqa':
            return {
                "id": _line['id'],
                "title": "",
                "context": "",
                "table": {
                    "id": _line['id'],
                    "header": _line['table_header'],
                    "rows": _line['table_data'],
                    "caption": "",
                },
                # "query": _line["question"],
                'query': ' '.join(_line["question_and_history"]),
                "label": _line["answer_text"],
            }
            

    def table2db(self, db_con: str, _line: dict):
        normalized = self.normalize_table(_line)
        df = pd.DataFrame(columns=[normalize_col_name(c)
                          for c in normalized['table']['header']])
        for ind, r in enumerate(normalized['table']['rows']):
            df.loc[ind] = r
        df.to_sql(name=self.table_name + '_' +
                  normalized['id'], con=db_con, if_exists='replace', index=False)

        print(f'Table {self.table_name} transformed in database ')
