from data_loader import TableFormat, TableLoader
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate


def get_k_shot_with_summary(k: int = 2):
    table_loader = TableLoader(table_name='tabfact', split='validation', use_sample=True)

    inds = [0, 2]
    summary_examples = ['The columns in the table are "wind farm, scheduled, capacity (mw), turbines, type, and location." The rows in the table represent different wind farms, with information about their scheduled dates, capacity, number of turbines, type, and location.',
                        "The table provides information about different events, including their names, establishment years, categories, subcategories, and main venues."]
    Output_examples = ['wind farm<DELETE> scheduled<KEEP> capacity (mw)<DELETE> turbines<KEEP> type<KEEP> location<KEEP>',
                       'event name<KEEP> established<KEEP> category<KEEP> sub category<DELETE> main venue<DELETE>']
    examples_prompt = PromptTemplate(input_variables=["table", "claim", "summary", "output"], template="""
    Table: {table}
    Claim: {claim}
    Summary: {summary}
    Output: {output}""")
    examples_dict = [{"table": TableFormat(format='none', data=table_loader.dataset[inds[i]], use_sampling=True).format_html(),
                      "claim": table_loader.dataset[inds[i]]['statement'],
                      "summary": summary_examples[i],
                      "output": Output_examples[i]} for i in range(k)]
    prompt_template = FewShotPromptTemplate(
        examples=examples_dict,
        example_prompt=examples_prompt,
        prefix=
    """
    Instruction: Given the following table and claim, let's first summarize the contents of the table, and then output the operations corresponding to each column which can help us judging the truth or falsity of claim.
    Operations: DELETE, KEEP, GROUP BY, COUNT, AVG, SUM, MAX, MIN, ORDER BY""",
        suffix="""
    Table: {table}
    Claim: {claim}
        """,
        input_variables=["table", "claim"],
    )
    return prompt_template

def get_k_shot(k: int=2):
    table_loader = TableLoader(table_name='tabfact', split='validation', use_sample=True)

    inds = [0, 2]
    Output_examples = ['wind farm<DELETE> scheduled<KEEP> capacity (mw)<DELETE> turbines<KEEP> type<KEEP> location<KEEP>',
                       'event name<KEEP> established<KEEP> category<KEEP> sub category<DELETE> main venue<DELETE>']
    examples_prompt = PromptTemplate(input_variables=["table", "claim", "output"], template=
    """
    Table: {table}
    Claim: {claim}
    Output: {output}""")
    num_k = 2
    examples_dict = [{"table": TableFormat(format='none', data=table_loader.dataset[inds[i]], use_sampling=True).format_html(),
                                        "claim": table_loader.dataset[inds[i]]['statement'],
                                        # "summary": summary_examples[i],
                                        "output": Output_examples[i]} for i in range(num_k)]
    prompt_template = FewShotPromptTemplate(
        examples=examples_dict,
        example_prompt=examples_prompt,
        prefix="""You are a brilliant table executor with the capabilities information retrieval, table parsing, table partition and semantic understanding who can understand the structural information of the table.
    Instruction: Given the following table and claim, you will output the operations corresponding to each column which can help us judging the truth or falsity of claim.
    Operations: DELETE(delete column unrelevant to the claim), KEEP(keep column relevant to the claim), GROUP BY(combine aggregate functions and group the result set by one or more columns), COUNT(returns the number of rows in column), AVG(returns the average value of a numeric column), SUM(returns the sum of a numeric column), MAX(returns the max value of a numeric column), MIN(returns the min value of a numeric column), ORDER BY(sort the value in ascending order)""",
        suffix=
        """
    Table: {table}
    Claim: {claim}
        """,
        input_variables=["table", "claim"],
)
    return prompt_template


def get_k_shot_with_answer(k: int=1):
    sqls = ["SELECT MIN(points) FROM DF WHERE rider = 'roger dutton / tony wright';"]
    thoughts = ["Based on the SQL query provided, the minimum number of points that Roger Dutton / Tony Wright received in the 1972 Isle of Man TT event was 3. Therefore, the claim that 2 is the fewest points they received is false. The output should be 0."]
    tables = ["<table>\n<caption>1972 isle of man tt</caption>\n<thead>\n<tr><th>  MIN(points)</th></tr>\n</thead>\n<tbody>\n<tr><td>3            </td></tr>\n</tbody>\n</table>"]
    claims = ["2 be the fewest point that roger dutton / tony wright receive"]
    # inds from test split
    examples_prompt = PromptTemplate(input_variables=["SQL", "table", "claim", "thought", "output"], template=
    """
    SQL Excuted: 
    ```{SQL}```
    Sub-table: {table}
    Query: {claim}
    Thought: {thought}
    Output: {output}
    """)
    examples_dict = dict(zip(["SQL", "table", "claim", "thought", "output"], [sqls[0], tables[0], claims[0], thoughts[0], '0']))
    prompt_template = FewShotPromptTemplate(
        examples=[examples_dict],
        example_prompt=examples_prompt,
        prefix="""Below is a sub-table generated by excuting the SQL. You need to understand the logic behind the SQL filtering and use the final subset to verify whether the provided claim/query is true or false, return 0 if it's false, or 1 if it's true. Please think step by step and only return 0 or 1 without any other information at last.""",
        suffix=
        """
    SQL Excuted: 
    ```{SQL}```
    Sub-table: {table}
    Query: {claim}
    Thought: """,
        input_variables=["table", "claim", "SQL"],
)
    return prompt_template
    
    
def get_k_shot_with_aug(k: int=2):
    table_loader = TableLoader(table_name='tabfact', split='validation', use_sample=True, small_test=False)

    inds = [3, 6]
    Output_examples = [
        # 'scheduled<KEEP> turbines<KEEP> type<KEEP> location<KEEP>',
                    #    'event name<KEEP> established<KEEP> category<KEEP> sub category<DELETE> main venue<DELETE>',
                       'team, goals for',
                       'year, game, platform (s)']
    examples_prompt = PromptTemplate(input_variables=["table", "claim", "output"], template=
    """
    Table: {table}
    Claim: {claim}
    Columns: {output}""")
    num_k = 2
    examples_dict = [{"table": TableFormat(format='none', data=table_loader.dataset[inds[i]], use_sampling=True).format_html(table_loader.dataset[inds[i]]['table']['caption']),
                                        "claim": table_loader.dataset[inds[i]]['statement'],
                                        # "summary": summary_examples[i],
                                        "output": Output_examples[i]} for i in range(num_k)]
    prompt_template = FewShotPromptTemplate(
        examples=examples_dict,
        example_prompt=examples_prompt,
        prefix="""
    You are a brilliant table executor with the capabilities information retrieval, table parsing, table partition and semantic understanding who can understand the structural information of the table.
    Given the following table and query, you should output columns related to the query or contain useful information about the query.""",
        suffix=
        """
    Table: {table}
    Claim: {claim}
    Extra information: {aug}
    Columns: """,
        input_variables=["table", "claim", "aug"],
)
    return prompt_template


def get_k_shot_with_aug_org(k: int=2):
    table_loader = TableLoader(table_name='tabfact', split='validation', use_sample=True, small_test=False)

    inds = [0, 2]
    Output_examples = ['wind farm<DELETE> scheduled<KEEP> capacity (mw)<DELETE> turbines<KEEP> type<KEEP> location<KEEP>',
                       'event name<KEEP> established<KEEP> category<KEEP> sub category<DELETE> main venue<DELETE>']
    examples_prompt = PromptTemplate(input_variables=["table", "claim", "output"], template=
    """
    Table: {table}
    Claim: {claim}
    Output: {output}""")
    num_k = 2
    examples_dict = [{"table": TableFormat(format='none', data=table_loader.dataset[inds[i]], use_sampling=True).format_html(),
                                        "claim": table_loader.dataset[inds[i]]['statement'],
                                        # "summary": summary_examples[i],
                                        "output": Output_examples[i]} for i in range(num_k)]
    prompt_template = FewShotPromptTemplate(
        examples=examples_dict,
        example_prompt=examples_prompt,
        prefix="""You are a brilliant table executor with the capabilities information retrieval, table parsing, table partition and semantic understanding who can understand the structural information of the table.
    Instruction: Given the following table and claim, you will output the operations corresponding to each column which can help us judging the truth or falsity of claim.
    Operations: DELETE(delete column unrelevant to the claim), KEEP(keep column relevant to the claim), GROUP BY(combine aggregate functions and group the result set by one or more columns), COUNT(returns the number of rows in column), AVG(returns the average value of a numeric column), SUM(returns the sum of a numeric column), MAX(returns the max value of a numeric column), MIN(returns the min value of a numeric column), ORDER BY(sort the value in ascending order)""",
        suffix=
        """
    Table: {table}
    Claim: {claim}
    Extra information: {aug}
        """,
        input_variables=["table", "claim", "aug"],
)
    return prompt_template
    
    
    
    
    
