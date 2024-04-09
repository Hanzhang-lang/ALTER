from sqlalchemy import create_engine
from executor import SQLManager
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI, OpenAI
from data_loader import TableLoader, TableFormat, TableAug
from prompt_manager import get_k_shot_with_aug, row_instruction, answer_instruction, get_k_shot_with_answer
import json
import os
from utils import eval_fv_match
import logging
import datetime
from typing import List
from tqdm import tqdm
import pandas as pd
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)
# logger.addHandler(handler)



def save_json(table_names: List, label_list: List, pred_list: List,  file_path):
    """
    Save specific list items in json format
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    assert len(table_names) == len(label_list)
    data_list = []
    for i in range(len(table_names)):
        data_list.append({
            'table_name': table_names[i],
            'label': label_list[i],
            'pred': pred_list[i]
        })
    with open(file_path, 'w') as file:
        for item in data_list:
            json_string = json.dumps(item)
            file.write(json_string + '\n')


def split_answer(answer: str):
    specail_tokens = ['Output']
    for s in specail_tokens:
        if s in answer:
            parts = answer.split(s)
            break
    summary = parts[0].split(':')[1].strip()
    operations = parts[1].split(':')[1].strip()
    return summary, operations



def pipeline(task_name: str,
             split: str,
             use_sample: bool,
             model_name: str):
    model = ChatOpenAI(model_name=model_name, openai_api_base="https://api.chatanywhere.tech/v1",
                       openai_api_key="sk-kxgtm71G6zwC44lglIF5CfiEVVzjjc39TOtppkNAwrVA2fUW")
    engine = create_engine('sqlite:///db/sqlite/tabfact.db', echo=False)
    
    manager = SQLManager(engine=engine)
    small_test = True
    table_loader = TableLoader(
        table_name=task_name, split=split, use_sample=use_sample, small_test=small_test)
    table_aug = TableAug(model)
    num_samples = len(table_loader.dataset)
    batch_size = 16
    num_batches = num_samples // batch_size
    batches_SQL = []

    execute = False
    verbose = True
    save_file = True
    load_sql = False 
    # stage_1: column pick up   stage_2: SQL Generate  stage_3: Answer output
    stage_1 = False
    stage_2 = False
    stage_3 = True
    # get k_shot example
    preds, ground, table_names = [], [], []
    if stage_1:
        k_shot_prompt = get_k_shot_with_aug()
        save_path = f"result/data/{task_name}_{split}_{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.json"
    elif stage_2:
        k_shot_prompt = row_instruction
        logger.info('\n' + k_shot_prompt.format(table='test-table',
                claim='test-claim', aug='test-aug'))
        save_path = f"result/SQL/{task_name}_{split}_{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.json"
    elif stage_3:
        k_shot_prompt = get_k_shot_with_answer()
        save_path = f"result/answer/{task_name}_{split}_{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.json"
    llm_chain = LLMChain(llm=model, prompt=k_shot_prompt, verbose=verbose)
    
    aug_information = pd.read_csv(f"result/aug/{task_name}_{split}_summary.csv", index_col='table_id')
    with tqdm(
        total=num_batches + (1 if num_samples % batch_size > 0 else 0),
        desc=f"Processing {task_name}",
        ncols=150,
    ) as pbar:
        # for batch_num in range(num_batches + (1 if num_samples % batch_size > 0 else 0)):
        for batch_num in range(num_batches+ (1 if num_samples % batch_size > 0 else 0)):
            
            inputs = []
            start = batch_num * batch_size
            if start + batch_size >= num_samples:
                batch_size = num_samples - start            
            for i in range(batch_size):
                normalized_sample = table_loader.normalize_table(
                    table_loader.dataset[start + i])
                # 输入 table, claim
                table_names.append(normalized_sample['id'])
                ground.append(normalized_sample['query'])
                formatter = TableFormat(
                    format='none', data=normalized_sample, use_sampling=True)
                if stage_2:
                    #注意函数命名
                    #TODO: 修改内外循环逻辑，避免重复读取文件
                    sql_preds = []
                    with open('./result/data/tabfact_test_04-09_06-19-47.json', 'r') as f:
                        lines = f.readlines()
                        for l in lines:
                            sql_preds.append(json.loads(l)['pred'])
                    columns = [formatter.normalize_col_name(c.strip()) for c in sql_preds[start + i].split(',')]
                    formatter.data = formatter.data.loc[:, columns]
                    
                if stage_3:
                    #注意函数命名
                    stage2_sql = []
                    with open('./result/SQL/tabfact_test_04-09_06-45-20.json', 'r') as f:
                        lines = f.readlines()
                        for l in lines:
                            stage2_sql.append(json.loads(l)['pred'])
                    #TODO:修改Format赋值逻辑
                    #TODO: 如果SQL执行报错的话，如何处理
                    try:
                        formatter.data = manager.execute_from_df(stage2_sql[start + i], formatter.all_data, table_name='DF')
                    except:
                        stage2_sql[start + i] = 'no SQL execution'
                    inputs.append(dict({'table': formatter.format_html(table_caption=normalized_sample['table']['caption']),
                                    'claim': normalized_sample['query'],
                                    'SQL':  stage2_sql[start + i],
                                    }))
                if not stage_3:      
                    summary_aug, column_aug = aug_information.loc[normalized_sample['id']]['summary'], aug_information.loc[normalized_sample['id']]['column_description'] 
                    
                    inputs.append(dict({'table': formatter.format_html(table_caption=normalized_sample['table']['caption']),
                                        'claim': normalized_sample['query'],
                                        'aug':  summary_aug + table_aug.table_size(formatter) + f'\ncolumn info: {column_aug}'
                                        }))
                if verbose:
                    logger.info(f'Table-id: {start + i}')
            # call llm to get batch executable sql
            batch_pred = llm_chain.batch(inputs, return_only_outputs=True)
            
            preds.extend([pred['text'] for pred in batch_pred])
            pbar.update(1)
    if save_file:
        save_json(table_names, ground, preds, save_path)
    
    evaluate = True
    if evaluate:
        labels = table_loader.dataset['label']
        preds = []
        with open('./result/SQL/tabfact_test_04-09_06-45-20.json', 'r') as f:
            lines = f.readlines()
            for l in lines:
                preds.append(json.loads(l)['pred'].split(':')[1].strip())


        # do evaluation
        accuracy = eval_fv_match(preds, labels)
        logger.info(f'Evaluate end, Accuracy: {accuracy}')


if __name__ == "__main__":
    pipeline()
