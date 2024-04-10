import json
from data_loader import TableFormat
from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAI
from utils import parse_output
import logging
from langchain.chains import LLMChain
from langchain_community.callbacks import get_openai_callback
logger = logging.getLogger(__name__)
# with open('data_loader/small_test_id.json', 'r') as f:
#     small_test_id = json.load(f)


class TableAug:
    def __init__(self, model=None) -> None:
        self.batch_method_mapping = {
            "summary": self.batch_summary_aug, "schema": self.batch_schema_aug}
        if model:
            self.llm = model
        else:
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", openai_api_base="https://api.chatanywhere.cn/v1",
                                  openai_api_key="sk-kxgtm71G6zwC44lglIF5CfiEVVzjjc39TOtppkNAwrVA2fUW")

    def schema_aug(self, formatter: TableFormat):
        pre_instruction = PromptTemplate(input_variables=["table"], template="""
        Instruction: Given the following table, you will add Metadata about the columns in the table.
        Metadata includes:
        - Numerical: whether the column content is numeric type like int or float.
        - Char: whether the column content is a text or description.
        - Date: whether the column content is datetime.

        You need to output all the column names with metadata in angle brackets.
        Example: name<Char> launched<Date> count<Numerical>

        Table: {table}
        Output:
        """)
        #
        output = self.llm.invoke([HumanMessage(
            content=pre_instruction.format(table=formatter.format_html()))]).content
        return parse_output(output)

    def batch_schema_aug(self, formatter: TableFormat, batch_data, batch_size: int, output_token=False):
        """
        batch schema augmentation
        """
        pre_instruction_schema = PromptTemplate(input_variables=["table"], template="""
        Instruction: Given the following table, you will add Metadata about the columns in the table.
        Metadata includes:
        - Numerical: whether the column content is numeric type like int or float.
        - Char: whether the column content is a text or description.
        - Date: whether the column content is datetime.

        You need to output all the column names with metadata in angle brackets.
        Example: name<Char> launched<Date> count<Numerical>

        Table: {table}
        Output:
        """)
        schema_list = []
        llm_chain = LLMChain(
            llm=self.llm, prompt=pre_instruction_schema, verbose=False)
        with get_openai_callback() as cb:
            # add
            batch_pred = llm_chain.batch([formatter.load_data_from_dic(batch_data[i]).format_html(
                batch_data[i]['caption']) for i in range(batch_size)], return_only_outputs=True)

        for i in range(len(batch_pred)):
            parts = batch_pred[i]['text']
            schema_list.append(parts)
        if output_token:
            logger.info(f"Batch Schema Augmentaion Tokens: {cb.total_tokens}")
        return schema_list

    def batch_summary_aug(self, formatter: TableFormat, batch_data, batch_size: int, output_token=False):
        """
        batch summary data
        """
        pre_instruction_summary = PromptTemplate(input_variables=['table'], template="""
        Instruction: Given the following table, you need to first summarize the contents of the table, then based on the summay, give a concluded description to each of the column.
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
            batch_pred = llm_chain.batch([formatter.load_data_from_dic(batch_data[i]).format_html(
                batch_data[i]['caption']) for i in range(batch_size)], return_only_outputs=True)

        for i in range(len(batch_pred)):
            parts = batch_pred[i]['text'].split('column description')
            summary_list.append(parts[0].split(':')[1].strip())
            description_list.append(parts[1].split(':')[1].strip())
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

    def batch_composition_aug(self, formatter: TableFormat, batch_data, batch_size: int, output_token=False):
        """
        batch composition augmentation
        """
        pre_instruction_com = PromptTemplate(input_variables=["table"], template="""
        Below is a subtable with columns filtered, you are required to infer the data distribution and format from the sample data of the sub-table.
        sub-table: {table}
        Refine commonalities about the structure within each table column.
        """)
        com_list = []
        llm_chain = LLMChain(
            llm=self.llm, prompt=pre_instruction_com, verbose=False)
        with get_openai_callback() as cb:
            # add
            batch_pred = llm_chain.batch([formatter.load_data_from_dic(batch_data[i]).format_html(
                batch_data[i]['caption']) for i in range(batch_size)], return_only_outputs=True)

        for i in range(len(batch_pred)):
            parts = batch_pred[i]['text']
            com_list.append(parts)
        if output_token:
            logger.info(
                f"Batch Composition Augmentaion Tokens: {cb.total_tokens}")
        return com_list

    def composition_aug(self, formatter: TableFormat, output_token=False):

        pre_instruction = PromptTemplate(input_variables=["table"], template="""
    Below is a subtable with columns filtered, you are required to infer the data distribution and format from the sample data of the sub-table.
    sub-table: {table}
    Refine commonalities about the structure within each table column.
    """)
        output = self.llm.invoke([HumanMessage(
            content=pre_instruction.format(table=formatter.format_html()))]).content

        return output

    def table_size(self, formatter: TableFormat):
        return f'The full table has {formatter.all_data.shape[0]} rows and {formatter.all_data.shape[1]} columns.'
