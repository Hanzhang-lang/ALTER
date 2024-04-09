from langchain_openai import ChatOpenAI, OpenAI
from data_loader import TableLoader, TableFormat, TableAug
import os
import sys
import pandas as pd
import logging
import datetime
from typing import List
from tqdm import tqdm
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger(__name__)



def save_csv(input_list: List[List], label_list: List, file_path):
    import pandas as pd
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    assert len(input_list) == len(label_list)
    df = pd.DataFrame()
    for i in range(len(label_list)):
        df[label_list[i]] = pd.Series(input_list[i])
    if os.path.exists(file_path) and file_path.endswith('.csv'):
        df_origin = pd.read_csv(file_path)
        df = pd.concat([df_origin, df], axis=0)
    df.to_csv(file_path, index=False, encoding='utf-8')


def augmentation(task_name: str,
             split: str,
             use_sample: bool,
             model_name: str,
             aug_type: str = 'summary',
             batch_size: int = 32):
    model = ChatOpenAI(model_name=model_name, openai_api_base="https://api.chatanywhere.cn/v1",
                       openai_api_key="sk-kxgtm71G6zwC44lglIF5CfiEVVzjjc39TOtppkNAwrVA2fUW")
    small_test = True
    table_loader = TableLoader(
        table_name=task_name, split=split, use_sample=use_sample, small_test=small_test)
    table_aug = TableAug(model)
    num_samples = len(table_loader.dataset)
    num_batches = num_samples // batch_size
    aug_path = f"result/aug/{task_name}_{split}_{aug_type}.csv"
    with tqdm(
        total=num_batches + (1 if num_samples % batch_size > 0 else 0),
        desc=f"Augmentation for {task_name}, AUG TYPE: {aug_type}",
        ncols=150,
    ) as pbar:
        for batch_num in range(num_batches + (1 if num_samples % batch_size > 0 else 0)):
            aug_tables = []
            start = batch_num * batch_size
            if start + batch_size >= num_samples:
                batch_size = num_samples - start + 1
            batch_data = table_loader.dataset[start: start+batch_size]
            
            if os.path.exists(aug_path) and aug_path.endswith('.csv'):
                auged_names = pd.read_csv(aug_path)['table_id']
            else: 
                auged_names = []
            table_names = []
            for i in range(batch_size):
                if batch_data['table'][i]['id'] not in auged_names:
                    auged_names.append(batch_data['table'][i]['id'])
                    aug_tables.append(batch_data['table'][i])            
            # 针对augmentation的cache
                    table_names.append(batch_data['table'][i]['id'])
            formatter = TableFormat(format='none')
            summary_augs, column_augs = table_aug.batch_summary_aug(
                formatter, aug_tables, len(table_names), output_token=True)
            save_csv([summary_augs, column_augs, table_names], [
                        'summary', 'column_description', 'table_id'], aug_path)
            pbar.update(1)
                
                
            


if __name__ == "__main__":
    augmentation('tabfact', 'test', True, 'gpt-3.5-turbo-0125')
