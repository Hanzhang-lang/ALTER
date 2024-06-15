from sqlalchemy import create_engine
from executor import SQLManager
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI, OpenAI
from data_loader import TableLoader, TableFormat, TableAug
from prompt_manager import get_k_shot_with_aug, row_instruction, answer_instruction, view_instruction, get_k_shot_with_schema_linking
import json
import os
from utils import eval_fv_match, normalize_schema, parse_specific_composition, parse_output
import logging
import datetime
from typing import List
from tqdm import tqdm
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_fixed, retry_if_exception_type
from langchain_community.llms.openai import completion_with_retry
import openai
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
             model_name: str,
             small_test: bool,
             stage_1: bool,
             stage_2: bool,
             stage_3: bool,
             batch_size: int):
    model = ChatOpenAI(model_name=model_name, openai_api_base="https://api.chatanywhere.tech/v1",
                       openai_api_key="sk-kxgtm71G6zwC44lglIF5CfiEVVzjjc39TOtppkNAwrVA2fUW", temperature=0.1, max_retries=5, request_timeout=600)
    engine = create_engine('sqlite:///db/sqlite/tabfact.db', echo=False)
    manager = SQLManager(engine=engine)
    table_loader = TableLoader(
        table_name=task_name, split=split, use_sample=use_sample, small_test=small_test)
    # num_samples = len(table_loader.dataset)
    num_samples = 255
    num_batches = num_samples // batch_size
    verbose = False
    save_file = True
    use_schema = True
    use_composition = True
    # stage_1: column pick up   stage_2: SQL Generate  stage_3: SQL Excution and Answer output
    # get k_shot example
    preds, ground, table_names = [], [], []
    if stage_1:
        k_shot_prompt = get_k_shot_with_schema_linking()
        save_path = f"result/data/{task_name}_{split}_{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.json"
    elif stage_2:
        k_shot_prompt = row_instruction
        # k_shot_prompt = view_instruction
        logger.info('\n' + k_shot_prompt.format(table='test-table',
                claim='test-claim', aug='test-aug'))
        save_path = f"result/SQL/{task_name}_{split}_{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.json"
        stage1_column = []
        with open('./result/data/tabfact_test_04-18_14-13-13.json', 'r') as f:
            lines = f.readlines()
            for l in lines:
                stage1_column.append(json.loads(l)['pred'])
    elif stage_3:
        k_shot_prompt = get_k_shot_with_answer()
        # k_shot_prompt = answer_instruction
        save_path = f"result/answer/{task_name}_{split}_{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.json"
        stage2_sql = []
        with open('./result/SQL/tabfact_test_04-20_02-04-20.json', 'r') as f:
            lines = f.readlines()
            for l in lines:
                stage2_sql.append(json.loads(l)['pred'])
    llm_chain = LLMChain(llm=model, prompt=k_shot_prompt, verbose=verbose)
    
    aug_information = pd.read_csv(f"result/aug/{task_name}_{split}_summary.csv", index_col='table_id')
    schema_information = []
    composition_information = []
    if use_schema:
        schema_information = pd.read_csv(f"result/aug/{task_name}_{split}_schema.csv", index_col='table_id')
    if use_composition:
        composition_information = pd.read_csv(f"result/aug/{task_name}_{split}_composition.csv", index_col='table_id')
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
                    columns = [formatter.normalize_col_name(c.strip()) for c in stage1_column[start + i].split(',')]
                    if use_schema:
                        formatter.normalize_schema(schema_information.loc[normalized_sample['id']]['schema'])
                    try:
                        formatter.data = formatter.data.loc[:, columns]
                    except:
                        pass
                    
                if stage_3:
                    #注意函数命名
                    #TODO:修改Format赋值逻辑
                    #TODO: 如果SQL执行报错的话，如何处理
                    try:
                        # before execute sql, normalize data format
                        #TODO:思考，是否只在执行前进行normalize
                        # formatter.data = normalize_table(formatter.data, schema_information.loc[normalized_sample['id']]['schema'])
                        formatter.normalize_schema(schema_information.loc[normalized_sample['id']]['schema'])
                        formatter.data = manager.execute_from_df(stage2_sql[start + i], formatter.all_data, table_name='DF')
                    except:
                        stage2_sql[start + i] = 'no SQL execution'
                        formatter.data = formatter.all_data
                    inputs.append(dict({'table': formatter.format_html(table_caption=normalized_sample['table']['caption']),
                                    'claim': normalized_sample['query'],
                                    'SQL':  stage2_sql[start + i],
                                    }))
                if not stage_3:      
                    summary_aug, column_aug = aug_information.loc[normalized_sample['id']]['summary'], aug_information.loc[normalized_sample['id']]['column_description'] 
                    # 区分stage_2和stage_1的augmentation info
                    if stage_1:
                        col_names, col_infos = parse_output(column_aug, pattern=r'([^<]*)<([^>]*)>')
                        # _, compositions = parse_output(composition_information.loc[normalized_sample['id']]['composition'], pattern = r'\d. (.+?): (.+)')
                        extra_col_info = []
                        for i_c in range(len(col_names)):
                            extra_col_info.append(f'{i_c + 1}. {col_names[i_c]}: {col_infos[i_c]}')
                        extra_information = summary_aug + '\n' + '\n'.join(extra_col_info)
                    if stage_2:
                        extra_information = '\n'.join(parse_specific_composition(composition_information.loc[normalized_sample['id']]['composition'], formatter.data.columns))
                    inputs.append(dict({'table': formatter.format_html(table_caption=normalized_sample['table']['caption']),
                                        'claim': normalized_sample['query'],
                                        'aug':  extra_information
                                        }))
                if verbose:
                    logger.info(f'Table-id: {start + i}')
            # call llm to get batch executable sql

            batch_pred = llm_chain.batch(inputs, return_only_outputs=True)
            
            preds.extend([pred['text'] for pred in batch_pred])
            pbar.update(1)
    if save_file:
        save_json(table_names, ground, preds, save_path)
    
    evaluate = False
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
