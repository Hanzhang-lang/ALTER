from sqlalchemy import create_engine
from executor import SQLManager
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.callbacks import get_openai_callback
from data_loader import TableLoader, TableFormat, TableAug
from prompt_manager import get_step_back_prompt, get_decompose_prompt, row_instruction, get_k_shot_with_schema_linking, extra_answer_instruction, muilti_answer_instruction, get_k_shot_with_aug
from prompt_manager import get_decompose_prompt_wiki, get_step_back_prompt_wiki, get_k_shot_with_aug_wiki, get_k_shot_with_answer_wiki
import json
import os
from utils import parse_specific_composition_zh, parse_output, add_row_number
import logging
import datetime
from typing import List
from tqdm import tqdm
import pandas as pd
import concurrent.futures
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
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

def pipeline(task_name: str,
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
        table_name=task_name, split=split, use_sample=use_sample, small_test=small_test, cache_dir='/media/disk2/datasets/')
    num_samples = len(table_loader.dataset)
    num_samples = 10
    num_batches = num_samples // batch_size
    token_count= []
    embeddings = HuggingFaceBgeEmbeddings(
            model_name='BAAI/bge-large-en',
            model_kwargs={'device': 'cuda:0', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True})
    save_path = f"result/answer/{task_name}_{split}_{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.csv"
        
    
    summary_information = pd.read_csv(f"result/aug/{task_name}_{split}_summary.csv", index_col='table_id')
    schema_information = pd.read_csv(f"result/aug/{task_name}_{split}_schema.csv", index_col='table_id')
    composition_information = pd.read_csv(f"result/aug/{task_name}_{split}_composition.csv", index_col='table_id')
    
    
    def scene_with_answer(query, sample, return_sub=False, verbose=verbose, k=3):
        formatter = TableFormat(format='none', data=sample, save_embedding=True, embeddings=embeddings)
        formatter.normalize_schema(schema_information.loc[sample['table']['id']]['schema'])
        if k == 0:
            sample_data = formatter.get_sample_data(sample_type='head', k=k)
        else:
            sample_data = formatter.get_sample_data(sample_type='embedding', query=query, k=k)
        with get_openai_callback() as cb:
            llm_chain = LLMChain(llm=model, prompt=get_k_shot_with_aug_wiki(), verbose=verbose)
            summary_aug, column_aug = summary_information.loc[sample['id']]['summary'], summary_information.loc[sample['id']]['column_description'] 
            col_names, col_infos = parse_output(column_aug, pattern=r'([^<]*)<([^>]*)>')
            extra_col_info = []
            for i_c in range(len(col_names)):
                extra_col_info.append(f'{i_c + 1}. {col_names[i_c]}: {col_infos[i_c]}')
            stage_1_batch_pred = llm_chain.batch([dict({'table': TableFormat.format_html(data=sample_data, table_caption=sample['table']['caption']),
                                            'claim': query,
                                            'aug':  summary_aug +'\n'+ '\n'.join(extra_col_info)
                                            })], return_only_outputs=True)[0]['text']
            if verbose:
                logger.info(stage_1_batch_pred)
            stage_1_batch_pred = stage_1_batch_pred.split(':')[-1]
            extra_cols = formatter.get_sample_column(embeddings, column_aug)
            columns = list(set([c.strip() for c in stage_1_batch_pred.split(',')] + extra_cols))
            # stage 2: SQL generation
            
            llm_chain = LLMChain(llm=model, prompt=row_instruction, verbose=verbose)
            try: 
            # formatter.all_data = formatter.all_data.loc[:, columns]
                sample_data = add_row_number(sample_data.loc[:, columns])
            except:
                sample_data = add_row_number(sample_data)
            extra_information = []
            tuples = parse_specific_composition_zh(composition_information.loc[sample['table']['id']]['composition'], sample_data.columns)
            for col, com in tuples:
                if len(pd.unique(formatter.all_data[col])) < 6:
                    com += f' (Values like {", ".join(list(formatter.all_data[col].dropna().unique().astype(str)))})'
                    extra_information.append(col + ':' + com)
                else:
                    com += f' (Values like {", ".join(list(formatter.all_data[col].dropna().unique()[:3].astype(str)))}...)'
                    extra_information.append(col + ':' + com)
            extra_information.append('row_number: row number in the original table')
            stage_2_batch_pred = llm_chain.batch([dict({'table': TableFormat.format_html(data = sample_data, table_caption=sample['table']['caption']),
                                            'claim': query,
                                            'aug':  summary_aug + '\nColumn information:\n' + '\n'.join(extra_information)
                                            })], return_only_outputs=True)[0]['text'].replace("–", "-").replace("—", "-").replace("―", "-").replace("−", "-")
        
            if verbose:
                logger.info(stage_2_batch_pred)
            if return_sub:   
            # stage 3: SQL Excution
                try: 
                    execute_data = manager.execute_from_df(stage_2_batch_pred, add_row_number(formatter.all_data), table_name='DF')
                except:
                    execute_data = formatter.all_data
                    stage_2_batch_pred = 'SELECT * from DF;'
                if len(execute_data) == 0:
                    return query, stage_2_batch_pred, 'No data from database', cb.total_tokens
                return query, stage_2_batch_pred, TableFormat.format_html(data=execute_data), cb.total_tokens
            
            else:
                llm_chain = LLMChain(llm=model, prompt=extra_answer_instruction, verbose=verbose)
                response = llm_chain.batch([dict({'table': TableFormat.format_html(execute_data),
                                                'claim': query,
                                                'SQL':  stage_2_batch_pred
                                                })], return_only_outputs=True)[0]['text']
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
                all_tokens = 0
                normalized_sample = table_loader.normalize_table(
                    table_loader.dataset[start + i])
                table_names.append(normalized_sample['id'])
                formatter = TableFormat(
                    format='none', data=normalized_sample,save_embedding=False)
                all_queries = []
                sample_data = formatter.get_sample_data(sample_type='random')
                with get_openai_callback() as cb:
                    llm_chain = LLMChain(llm=model, prompt=get_step_back_prompt_wiki(), verbose=False)
                    batch_pred = llm_chain.batch([{"query": normalized_sample['query'], "table": TableFormat.format_html(sample_data)}], return_only_outputs=True)
                    if batch_pred[0]['text'].strip() != normalized_sample['query']:
                        all_queries.append(batch_pred[0]['text'].strip())
                    llm_chain = LLMChain(llm=model, prompt=get_decompose_prompt_wiki(), verbose=False)
                    batch_pred = llm_chain.batch([{"query": normalized_sample['query'], "table": TableFormat.format_html(sample_data)}], return_only_outputs=True)
                    all_queries.extend([q.strip() for q in batch_pred[0]['text'].split(';')])
                all_queries = list(set(all_queries))
                args_list = [{"query": q, "sample": normalized_sample} for q in all_queries]
                ans_from_scene = parallel_run_kwargs(scene_with_answer, args_list) 
                scene_results =  [res[0] for res in ans_from_scene if 'Cannot get answer from sub-table' not in res[0] ]
                all_tokens += sum([res[1] for res in ans_from_scene])        
                with get_openai_callback() as cb:
                    imp_input = scene_with_answer(normalized_sample['query'], normalized_sample, return_SQL=True, verbose=False)
                    inputs.append({"query": normalized_sample['query'],"SQL": imp_input[1], "table": imp_input[2], "information": '\n'.join(scene_results)})
                all_tokens += cb.total_tokens
                extras.append('\n'.join(scene_results))
                token_count.append(all_tokens)
            llm_chain = LLMChain(llm=model, prompt=get_k_shot_with_answer_wiki(), verbose=True)
            batch_preds = llm_chain.batch(inputs, return_only_outputs=True)
            preds.extend([pred['text'] for pred in batch_preds])
            pbar.update(1)
            if save_file:
                save_csv([preds, token_count, extras], label_list=['preds', 'tokens', 'extra'], file_path=save_path)


if __name__ == "__main__":
    pipeline()
