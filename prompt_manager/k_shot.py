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

def get_k_shot_with_string(k: int = 2):
    Output_examples = [
        # """leagues_entering_at_this_round: different league name joint with '&' or None value""",
                       """
    goal: sequential number like 1, 2, 3...
    date: date in the format of Y-M-D
    venue: venue in the format of location, city, country
    score: score number in the format of X-Y
    result: result number in the format of X-Y
    competition: competition name or friendly
    nation: nation name with abbreviation within parentheses"""]
    examples_prompt = PromptTemplate(input_variables=["table", "claim", "summary", "output"], template="""
    Table: {table}
    Output: {example}""")
    table_loader = TableLoader(table_name='tabfact', split='validation', use_sample=False)
    example_data = TableFormat(format='none', data=table_loader.dataset[20], use_sampling=True).data.iloc[:, [0,1,2,3,5]].reset_index(drop=True)
    example_data['nation'] = TableFormat(format='none', data=table_loader.dataset[130], use_sampling=True).data.iloc[:, 1].reset_index(drop=True)
    examples_dict = [{"table": TableFormat(format='none', data=example_data, use_sampling=True).format_html(),
                      "example": Output_examples[i]} for i in range(k)]
    prompt_template = FewShotPromptTemplate(
        examples=examples_dict,
        example_prompt=examples_prompt,
        prefix=
    """Below is a subtable with rows sampled, your task is to summarize and synthesize each column in the table, identifying commonalities in the string representations, and ultimately output string format commanalities for each column.
    The example is below:""",
        suffix="""Table: {table}""",
        input_variables=["table"],
    )
    return prompt_template

def get_k_shot_with_schema_linking(k: int=2):
    table_loader = TableLoader(table_name='tabfact', split='validation', use_sample=True, small_test=False)

    inds = [3, 6]
    Output_examples = [
                       'team, goals_for',
                       'year, game, platform_s']
    linking_examples = ['the team -> team; the most goal for -> goals_for',
                        'gamecube -> platform_s; gamecube game -> game; the first 3 year -> year;'
    ]
    examples_prompt = PromptTemplate(input_variables=["table", "claim", "output", "linking"], template=
    """
    Table: {table}
    Claim: {claim}
    Schema linking: {linking}
    Columns: {output}""")
    num_k = 2
    examples_dict = [{"table": TableFormat(format='none', data=table_loader.dataset[inds[i]], use_sampling=True).format_html(table_loader.dataset[inds[i]]['table']['caption']),
                                        "claim": table_loader.dataset[inds[i]]['statement'],
                                        "linking": linking_examples[i],
                                        # "summary": summary_examples[i],
                                        "output": Output_examples[i]} for i in range(num_k)]
    prompt_template = FewShotPromptTemplate(
        examples=examples_dict,
        example_prompt=examples_prompt,
        prefix="""
    You are a brilliant table executor with the capabilities information retrieval, table parsing, table partition and semantic understanding who can understand the structural information of the table.
    Given the following table and query, you should output columns related to the query or contain useful information about the query. 
    Here are some examples:""",
        suffix=
        """
    Table: {table}
    Claim: {claim}
    Extra information: {aug}
    """,
        input_variables=["table", "claim", "aug"],
)
    return prompt_template


    # prompt_template = FewShotPromptTemplate(
    #     examples=examples_dict,
    #     example_prompt=examples_prompt,
    #     prefix="""
    # You are a brilliant table executor with the capabilities information retrieval, table parsing, table partition and semantic understanding who can understand the structural information of the table.
    # Given the following table and query, you should output columns related to the query. 
    # Here’s how you should approach this:
    # Schema linking: link the terms in the query to the columns in the table using format #term -> #column_name
    # Columns: output columns mentioned in the schema linking process.""",
    #     suffix=
    #     """
    # Table: {table}
    # Claim: {claim}
    # Extra information: {aug}
    # """,
    # )
    
    
    
    
    
