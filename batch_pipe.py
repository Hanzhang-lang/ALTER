from sqlalchemy import create_engine
from executor import SQLManager
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.callbacks import get_openai_callback
from data_loader import TableLoader, TableFormat, TableAug
from prompt_manager import get_step_back_prompt, get_decompose_prompt, row_instruction, get_k_shot_with_schema_linking, extra_answer_instruction, muilti_answer_instruction, get_k_shot_with_aug
import json
import os
from utils import eval_fv_match, normalize_schema, parse_specific_composition, parse_output
import logging
import datetime
from typing import List
from tqdm import tqdm
import pandas as pd
import concurrent.futures
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)


def parallel_run_kwargs(func, args_list):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda kwargs: func(**kwargs), args_list)
        return list(results)
    
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

def new_pipeline(task_name: str,
             split: str,
             use_sample: bool,
             model_name: str,
             small_test: bool,
             batch_size: int,
             verbose: bool,
             save_file: bool, 
             aug_type: List):
    model = ChatOpenAI(model_name=model_name, openai_api_base="https://api.chatanywhere.tech/v1",
                       openai_api_key="sk-kxgtm71G6zwC44lglIF5CfiEVVzjjc39TOtppkNAwrVA2fUW", temperature=0.01, max_retries=5, request_timeout=6000)
    engine = create_engine('sqlite:///db/sqlite/tabfact.db', echo=False)
    manager = SQLManager(engine=engine)
    table_loader = TableLoader(
        table_name=task_name, split=split, use_sample=use_sample, small_test=small_test)
    num_samples = len(table_loader.dataset)
    num_samples = 100
    num_batches = num_samples // batch_size
    token_count= []

    save_path = f"result/answer/{task_name}_{split}_{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.csv"
        
    
    summary_information = pd.read_csv(f"result/aug/{task_name}_{split}_summary.csv", index_col='table_id')
    schema_information = pd.read_csv(f"result/aug/{task_name}_{split}_schema.csv", index_col='table_id')
    composition_information = pd.read_csv(f"result/aug/{task_name}_{split}_composition.csv", index_col='table_id')
    
    
    def scene_with_answer(query, sample, return_SQL=False, verbose=verbose):
        formatter = TableFormat(format='none', data=sample, use_sampling=True)
        with get_openai_callback() as cb:
            llm_chain = LLMChain(llm=model, prompt=get_k_shot_with_aug(), verbose=verbose)
            summary_aug, column_aug = summary_information.loc[sample['id']]['summary'], summary_information.loc[sample['id']]['column_description'] 
            col_names, col_infos = parse_output(column_aug, pattern=r'([^<]*)<([^>]*)>')
            extra_col_info = []
            for i_c in range(len(col_names)):
                extra_col_info.append(f'{i_c + 1}. {col_names[i_c]}: {col_infos[i_c]}')
            stage_1_batch_pred = llm_chain.batch([dict({'table': formatter.format_nl_sep(table_caption=sample['table']['caption']),
                                                'claim': query,
                                                #may lead to bias
                                                # 'aug':  summary_aug + '\n'.join(extra_col_info)
                                                })], return_only_outputs=True)[0]['text']
            logger.info(stage_1_batch_pred)
            stage_1_batch_pred = stage_1_batch_pred.split(':')[-1]
            
            # stage 2: SQL generation
            
            llm_chain = LLMChain(llm=model, prompt=row_instruction, verbose=verbose)
            columns = [formatter.normalize_col_name(c.strip()) for c in stage_1_batch_pred.split(',')]
            
            try: 
                formatter.data = formatter.data.loc[:, columns]
            except:
                pass
            extra_information = '\n'.join(parse_specific_composition(composition_information.loc[sample['id']]['composition'], formatter.data.columns))
            formatter.normalize_schema(schema_information.loc[sample['id']]['schema'])
            stage_2_batch_pred = llm_chain.batch([dict({'table': formatter.format_html(table_caption=sample['table']['caption']),
                                                'claim': query,
                                                'aug':  summary_aug + '\n Column information: \n' + extra_information
                                                })], return_only_outputs=True)[0]['text']
        
            logger.info(stage_2_batch_pred)
            # stage 3: SQL Excution
            try: 
                formatter.data = manager.execute_from_df(stage_2_batch_pred, formatter.all_data, table_name='DF')
            except:
                formatter.data = formatter.all_data
                stage_2_batch_pred = 'SELECT * from DF;'
            
            if return_SQL:
                logger.info(cb.total_tokens)
                if len(formatter.data) == 0:
                    return query, stage_2_batch_pred, 'No data from database', cb.total_tokens
                return query, stage_2_batch_pred, formatter.format_html(), cb.total_tokens
            else:
                llm_chain = LLMChain(llm=model, prompt=extra_answer_instruction, verbose=verbose)
                response = llm_chain.batch([dict({'table': formatter.format_html(),
                                                        'claim': query,
                                                        'SQL':  stage_2_batch_pred
                                                        })], return_only_outputs=True)[0]['text']
                logger.info(cb.total_tokens)
                return response, cb.total_tokens
    
    with tqdm(
        total=num_batches + (1 if num_samples % batch_size > 0 else 0),
        desc=f"Processing {task_name}",
        ncols=150,
    ) as pbar:
        for batch_num in range(num_batches+ (1 if num_samples % batch_size > 0 else 0)):
            start = batch_num * batch_size
            if start + batch_size >= num_samples:
                batch_size = num_samples - start  
            inputs,extras, token_count, table_names, preds = [],[],[],[], []
            for i in range(batch_size):
                normalized_sample = table_loader.normalize_table(
                    table_loader.dataset[start + i])
                # Do query augmentation first
                table_names.append(normalized_sample['id'])
                formatter = TableFormat(
                    format='none', data=normalized_sample, use_sampling=True)
                all_queries = []
                llm_chain = LLMChain(llm=model, prompt=get_step_back_prompt(), verbose=False)
                batch_pred = llm_chain.batch([{"query": normalized_sample['query'], "table": formatter.format_html()}], return_only_outputs=True)
                all_queries.append(batch_pred[0]['text'].split(':')[-1])
                llm_chain = LLMChain(llm=model, prompt=get_decompose_prompt(), verbose=False)
                batch_pred = llm_chain.batch([{"query": normalized_sample['query'], "table": formatter.format_html()}], return_only_outputs=True)
                all_queries.extend(batch_pred[0]['text'].split(';'))
                
                args_list = [{"query": q, "sample": normalized_sample} for q in all_queries]
                ans_from_scene = parallel_run_kwargs(scene_with_answer, args_list) 
                scene_results = [res[0] for res in ans_from_scene if res[0] != 'Cannot get answer from sub-table']
                all_tokens = sum([res[1] for res in ans_from_scene])        
                with get_openai_callback() as cb:
                    imp_input = scene_with_answer(normalized_sample['query'], normalized_sample, return_SQL=True, verbose=False)
                    inputs.append({"query": normalized_sample['query'],"SQL": imp_input[1], "table": imp_input[2], "information": '\n'.join(scene_results)})
                extras.append('\n'.join(scene_results))
                token_count.append(all_tokens)
            llm_chain = LLMChain(llm=model, prompt=muilti_answer_instruction, verbose=True)
            batch_preds = llm_chain.batch(inputs, return_only_outputs=True)
            preds.extend([pred['text'] for pred in batch_preds])
            pbar.update(1)
            if save_file:
                save_csv([table_names, extras, preds, token_count], label_list=['table_name', 'extra_information', 'preds', 'token'], file_path=save_path)


if __name__ == "__main__":
    new_pipeline()
