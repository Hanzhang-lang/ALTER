from langchain.prompts.prompt import PromptTemplate

Zero_shot_prompt = PromptTemplate(input_variables=["table", "claim", "aug"], 
                                  template="""
Below is a subtable with columns filtered, you are required to infer the data distribution and format from the sample data of the sub-table. Carefully analyze the query, based on the augmentation information, you need to verify whether the provided claim/query are true or false. Return 0 if it's false, or 1 if it's true. Only return 0 or 1 without any other information. 
sub-table: {table}
Query: {claim}
Extra information: {aug}
Output: """
)


row_instruction = PromptTemplate(input_variables=["table", "claim", "aug"], 
                                 template="""
Our ultimate goal is to answer query based on the table. Below is a subtable with columns filtered, you are required to infer the data distribution and format from the sample data of the sub-table. Carefully analyze the query, based on the augmentation information, write an SQL statement using table DF that complete query.
sub-table: {table}
Query: {claim}
Extra information: {aug}
SQL: """)

answer_instruction = PromptTemplate(input_variables=["SQL", "table", "claim"], 
                                    template="""
Below is a sub-table generated by excuting the SQL. You need to understand the logic behind the SQL filtering and use the final subset to verify whether the provided claim/query is true or false, return 0 if it's false, or 1 if it's true. Please think step by step and only return 0 or 1 without any other information at last.
SQL Excuted: 
```{SQL}```
Sub-table: {table}
Query: {claim}
""")



# answer_instruction = PromptTemplate(input_variables=["table", "claim"], template=
# """
# Below is a sub-table generated by excuting the SQL.
# SQL Excuted:
# ```{SQL}```
# Sub-table: {table}

# You need to understand the logic behind the SQL filtering and use the final subset to verify whether the provided claim/query is true or false, return 0 if it's false, or 1 if it's true. Only return 0 or 1 without any other information.
# Output your thought below:
# Query: {claim}
# """)
