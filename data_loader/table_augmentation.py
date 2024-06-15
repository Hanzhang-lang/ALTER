import json
from data_loader import TableFormat
from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAI
from utils import parse_output
from prompt_manager import get_k_shot_with_string
import logging
from typing import List
from langchain.chains import LLMChain
from langchain_community.callbacks import get_openai_callback
logger = logging.getLogger(__name__)


class TableAug:
    def __init__(self, model=None, use_embedding=True) -> None:
        self.batch_method_mapping = {
            "summary": self.batch_summary_aug, "schema": self.batch_schema_aug}
        if model:
            self.llm = model
        else:
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                                  openai_api_key="sk-Kfk2WiZcPgajLVtdGPGmfahxCmWqJSbMeRck5sXujlMS4Nai")

    def schema_aug(self, formatter: TableFormat):
        pre_instruction = PromptTemplate(input_variables=["table"], template="""
Instruction: Given the following table, you will add schema type about the columns in the table.
Schema type includes:
- Numerical: consists of digits and numerical symbols like decimal points or signs.
- Char: whether column content is a phrase or description.
- Date: whether column content represents time or date.

        You need to output all the column names with schemas in angle brackets.
        Example: name<Char> launched<Date> count<Numerical>

        Table: {table}
        Output:
        """)
        output = self.llm.invoke([HumanMessage(
            content=pre_instruction.format(table=formatter.format_html()))]).content
        return parse_output(output)

    def batch_schema_aug(self, tables: List, captions: List, output_token=False):
        """
        batch schema augmentation
        """
        pre_instruction_schema = PromptTemplate(input_variables=["table"], template="""
Instruction: Given the following table, you will add schema type about the columns in the table.
Schema type includes:
- Numerical: consists of digits and numerical symbols like decimal points or signs.
- Char: whether column content is a phrase or description.
- Date: whether column content represents time or date.

        You need to output all the column names with schemas in angle brackets.
        Example: name<Char> launched<Date> count<Numerical>

        Table: {table}
        Output:
        """)
        schema_list = []
        llm_chain = LLMChain(
            llm=self.llm, prompt=pre_instruction_schema, verbose=False)
        with get_openai_callback() as cb:
            # add
            batch_pred = llm_chain.batch([TableFormat.format_html(
                data=tables[i], table_caption=captions[i]) for i in range(len(tables))], return_only_outputs=True)

        for i in range(len(batch_pred)):
            parts = batch_pred[i]['text']
            schema_list.append(parts)
        if output_token:
            logger.info(f"Batch Schema Augmentaion Tokens: {cb.total_tokens}")
        return schema_list

    def batch_sum_aug(self, tables: List, captions: List, output_token=False):
        """
        batch with table summary augmentation
        """
        pre_instruction_schema = PromptTemplate(input_variables=["table"], template="""
        Instruction: Given the following table, you need to summarize the contents of the table and tell what table is about.
        Table: {table}

        The output should use the following format: 
        Summary: #summary for table contents
        
        Summary:
        """)
        summary_list = []
        llm_chain = LLMChain(
            llm=self.llm, prompt=pre_instruction_schema, verbose=False)
        with get_openai_callback() as cb:
            # add
            batch_pred = llm_chain.batch([TableFormat.format_html(
                data=tables[i], table_caption=captions[i]) for i in range(len(tables))], return_only_outputs=True)
        for i in range(len(batch_pred)):
            parts = batch_pred[i]['text']
            summary_list.append(parts)
        if output_token:
            logger.info(f"Batch Summary Augmentaion Tokens: {cb.total_tokens}")
        return summary_list

    def batch_summary_aug(self, tables: List, captions: List, output_token=False):
        # TODO: split 2 parts
        """
        batch summary data
        """
        pre_instruction_summary = PromptTemplate(input_variables=['table'], template="""
        Instruction: Given the following table, you need to first summarize the contents of the table, then based on the summary, give a concluded description of each of the columns.
        Table: {table}

        The output should use the following format: 
        table summary: #summary for table contents
        column description: You need to output all the column names with description in angle brackets
        example: launched<The launched date for the competition> date<The date of the match>
        """)
        summary_list = []
        description_list = []
        llm_chain = LLMChain(
            llm=self.llm, prompt=pre_instruction_summary, verbose=False)
        with get_openai_callback() as cb:
            # add
            batch_pred = llm_chain.batch([TableFormat.format_html(
                data=tables[i], table_caption=captions[i]) for i in range(len(tables))], return_only_outputs=True)

        for i in range(len(batch_pred)):
            try:
                parts = batch_pred[i]['text'].split('column description')
                summary_list.append(parts[0].split(':')[1].strip())
                description_list.append(parts[1].split(':')[1].strip())
            except:
                summary_list.append("")
                description_list.append(batch_pred[i]['text'])
        if output_token:
            logger.info(f"Batch Summary Augmentaion Tokens: {cb.total_tokens}")
        return summary_list, description_list

    def summary_aug(self, formatter: TableFormat, output_token=False):
        """
        TODO: split it into two aug methods
        """
        pre_instruction_summary = PromptTemplate(input_variables=['table'], template="""
        Instruction: Given the following table, you need to first summarize the contents of the table, then based on the summay, give a concluded description to each of the column.
        Table: {table}

        The output should use the following format: 
        table summary: #summary for table contents
        column description: You need to output all the column names with description in angle brackets
        example: launched<The launched date for the competition> date<The date of the match>
        """)
        output = self.llm.invoke([HumanMessage(
            content=pre_instruction_summary.format(table=formatter.format_html()))])
        if output_token:
            logger.info(
                f"Summary Augmentation Token: {output.response_metadata['token_usage']}")
        specail_tokens = ['column description']
        for s in specail_tokens:
            if s in output.content:
                parts = output.content.split(s)
                break
        summary = parts[0].split(':')[1].strip()
        operations = parts[1].split(':')[1].strip()
        return summary, operations

    def batch_composition_aug(self, tables: List, captions: List, output_token=False):
        """
        batch composition augmentation
        """
        pre_instruction_com = PromptTemplate(input_variables=["table"], template="""
        Instruction: Below is a subtable with rows sampled, you are required to infer the data distribution and format from the sample data. Refine commonalities in literal representations within each table column.
        You need to output in the following format: 
        number. Column_name: Commonalities
        #example format
        1. championship: Names of golf tournaments are listed with some additional information (e.g., 's open, classic)

        sub-table: {table}
        """)
        com_list = []
        llm_chain = LLMChain(
            llm=self.llm, prompt=pre_instruction_com, verbose=False)
        with get_openai_callback() as cb:
            # add schema augmentaion info first
            batch_pred = llm_chain.batch([TableFormat.format_html(
                data=tables[i], table_caption=captions[i]) for i in range(len(tables))], return_only_outputs=True)
        for i in range(len(batch_pred)):
            parts = batch_pred[i]['text']
            com_list.append(parts)
        if output_token:
            logger.info(
                f"Batch Composition Augmentaion  All Tokens: {cb.total_tokens}")
        return com_list

    def batch_string_aug(self, tables: List, captions: List, output_token=False):
        """
        batch composition augmentation
        """
        com_list = []
        llm_chain = LLMChain(
            llm=self.llm, prompt=get_k_shot_with_string(), verbose=False)
        with get_openai_callback() as cb:
            # add schema augmentaion info first
            batch_pred = llm_chain.batch([TableFormat.format_html(
                data=tables[i], table_caption=captions[i]) for i in range(len(tables))], return_only_outputs=True)
        for i in range(len(batch_pred)):
            parts = batch_pred[i]['text']
            com_list.append(parts)
        if output_token:
            logger.info(
                f"Batch String Augmentaion  All Tokens: {cb.total_tokens}")
        return com_list

    def table_size(self, formatter: TableFormat):
        return f'The full table has {formatter.all_data.shape[0]} rows and {formatter.all_data.shape[1]} columns.'
