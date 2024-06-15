import re
import sqlparse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from typing import List, Any
from pandas import DataFrame
import pandas as pd


class SQLManager:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.session_factory = sessionmaker(bind=engine)
        self.session = self.get_session()

    def get_session(self):
        return self.session_factory()

    def __sql_parse(self, sql):
        sql = sql.strip()
        parsed = sqlparse.parse(sql)[0]
        sql_type = parsed.get_type()

        table_name = parsed.get_name()

        first_token = parsed.token_first(skip_ws=True, skip_cm=False)
        ttype = first_token.ttype
        print(
            f"SQL:{sql}, ttype:{ttype}, sql_type:{sql_type}, table:{table_name}"
        )
        return parsed, ttype, sql_type, table_name

    def _query(self, query: str, session, fetch: str = "all"):
        """Run a SQL query and return the results as a list of tuples.

        Args:
            query (str): SQL query to run
            fetch (str): fetch type
        """
        result: List[Any] = []

        print(f"Query[{query}]")
        if not query:
            return result
        cursor = session.execute(text(query))
        if cursor.returns_rows:
            if fetch == "all":
                result = cursor.fetchall()
            elif fetch == "one":
                result = [cursor.fetchone()]
            else:
                raise ValueError(
                    "Fetch parameter must be either 'one' or 'all'")
            field_names = tuple(i[0:] for i in cursor.keys())

            result.insert(0, field_names)
            return result

    def get_simple_fields(self, command):
        """Get column fields about specified table."""
        return self.session.execute(text(command))

    def run(self, command: str, fetch: str = "all") -> List:
        """Execute a SQL command and return a string representing the results."""
        if not command or len(command) < 0:
            return []
        parsed, ttype, sql_type, table_name = self.__sql_parse(command)
        if ttype == sqlparse.tokens.DML:
            if sql_type == "SELECT":
                return self._query(command, self.session, fetch)
        else:
            self.get_simple_fields(command)

    def execute_from_df(self, command: str, data: DataFrame,  table_name='DF'):
        data.to_sql(table_name, self.engine, if_exists='replace', index=False)
        subtable = pd.read_sql(command, self.engine)
        return subtable

    def normalize_col_name(self, col_name, illegal_chars={'.': '', ' ': '_',
                                                          '\\': '_',  '(': '',
                                                          ')': '', '?': '',
                                                          '\n': '_', '&': '',
                                                          ':': '_', '/': '_',
                                                          ',': '_', '-': '_',
                                                          'from': 'c_from',
                                                          '\'': '',
                                                          '%': 'percent',
                                                          '#': 'num'}):
        if len(col_name) == 0:
            return 'NULL_COL'
        for c in illegal_chars:
            col_name = col_name.replace(c, illegal_chars[c])
        col_name = re.sub('_+', '_', col_name)
        return col_name
