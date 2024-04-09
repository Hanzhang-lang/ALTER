from datasets import load_dataset
import pandas as pd
import re


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
    def __init__(self, table_name: str, split: str = None, use_sample: bool = True, small_test: bool = False) -> None:
        """

        """
        self.table_name = table_name
        self.dataset = self.load_table(use_sample, split=split)
        if small_test:
            self.dataset = self.dataset.filter(lambda example: example['small_test'])

    def load_table(self, use_sample: bool = True, split: str = None, ):
        dataset = load_dataset(
            f"data_loader/{self.table_name}.py", verification_mode="no_checks", cache_dir="/media/disk2/datasets")
        if split:
            dataset = dataset[split]
        if use_sample and len(dataset) > 300:
            shuffled_dataset = dataset.shuffle(seed=42)
            return shuffled_dataset.select(range(300))
        return dataset

    def normalize_table(self, _line: dict):

        if self.table_name == 'tabfact':
            if str(_line["label"]) == "0":
                label = "0"
            elif str(_line["label"]) == "1":
                label = "1"
            else:
                label = "2"
            return {
                "id": _line['table']['id'],
                "title": "",
                "context": "",
                "table": {
                    "header": _line['table']['header'],
                    "rows": _line['table']['rows'],
                    "caption": _line['table']['caption'],
                },
                "query": _line["statement"],
                "label": label,
            }


    def table2db(self, db_con: str, _line: dict):

        normalized = self.normalize_table(_line)
        df = pd.DataFrame(columns=[normalize_col_name(c)
                          for c in normalized['table']['header']])
        for ind, r in enumerate(normalized['table']['rows']):
            df.loc[ind] = r
        # print(' '.join(normalized['table']['header']) + '*************')
        df.to_sql(name=self.table_name + '_' +
                  normalized['id'], con=db_con, if_exists='replace', index=False)

        print(f'Table {self.table_name} transformed in database ')
