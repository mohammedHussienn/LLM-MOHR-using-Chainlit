from dotenv import load_dotenv
import pyodbc
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing import List, Any, TypedDict
import os
import pandas as pd
from typing_extensions import Annotated
import logging
import json
from datetime import datetime, date


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


load_dotenv()


class SQLResponse(TypedDict):
    """TypedDict for structuring SQL query responses.

    Attributes:
        sql_query (str): The SQL query string
        column_names (List[str]): List of column names from the query result
    """
    sql_query: str
    column_names: List[str]

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

class State(TypedDict, total=False):
    tenant_id: int
    question: str
    query: str
    result: str
    answer: str
    failed_queries: List[str]

class SchemaManager:
    """Parses and manages the database schema."""
    def __init__(self, schema_file_path):
        self.schema_file_path = schema_file_path
        self.schema = {}

    def parse_schema(self):
        """Parses the schema file into a structured format."""
        with open(self.schema_file_path, 'r') as f:
            schema_raw = f.readlines()

        current_table = None
        for line in schema_raw:
            line = line.strip()
            if not line:
                continue

            if ':' in line:  # New table detected
                # Extract table name before the colon
                current_table = line.split(':')[0].strip()
                self.schema[current_table] = []
                # Get the columns part after the colon
                columns_part = line.split(':', 1)[1].strip()
                if columns_part:
                    # Split by comma and clean up each column definition
                    columns = [col.strip() for col in columns_part.split(',')]
                    for col in columns:
                        if '(' in col:  # Column has type information
                            col_name = col.split('(')[0].strip()
                            self.schema[current_table].append(col_name)
            elif current_table and line:  # Additional columns for current table
                # Split by comma and clean up each column definition
                columns = [col.strip() for col in line.split(',')]
                for col in columns:
                    if '(' in col:  # Column has type information
                        col_name = col.split('(')[0].strip()
                        self.schema[current_table].append(col_name)

    def get_table_info(self):
        """Returns the schema as a dictionary."""
        return self.schema

    def get_columns(self, table_name):
        """Fetches columns for a specific table."""
        return self.schema.get(table_name, [])

class DatabaseBackend:
    """Handles database operations and LLM interactions for SQL queries.

    This class manages database connections, SQL queries, and language model interactions
    for different response modes (raw, informative, conversational).
    """

    def __init__(self, schema_file_path="schema.txt"):
        # Initialize temperature
        self.temperature = {
            'r': 0,
            'i': 0.4,
            'c': 0.9
        }

        self.llm_for_agent = ChatOpenAI(
            model_name="gpt-4o", 
            temperature=0,
        )

        # Initialize parsers
        self.parser = {
            'r': StrOutputParser(),
            'i': StrOutputParser(),
            'c': StrOutputParser()
        }
        
        # Initialize prompts
        self.prompts = {
            'r': PromptTemplate(
                template="""
                You are a data formatting assistant. Convert the following text into a structured JSON format for table display.
                
                Original Question: {input}
                
                IMPORTANT FORMATTING RULES:
                1. Group related data together (e.g., all information about one employee should be in one row)
                2. Use consistent column names across all entries
                3. Convert ALL percentages to proper decimals (e.g., 22.46% becomes 0.2246)
                4. Keep numbers as numbers (not strings)
                5. Do not include any % symbols in the values
                6. Each employee should have one entry with all their information
                
                Example of CORRECT FORMAT this is just an example of what is needed. DO NOT FORGET THIS:
                {{
                    "table_data": [
                        {{
                            "column_name": "id",
                            "column_value": [1277, 1278, 1279, 1280]
                        }},
                        {{
                            "column_name": "name",
                            "column_value": ["John Smith", "Jane Doe", "Bob Wilson", "Alice Brown"]
                        }},
                        {{
                            "column_name": "shift_date",
                            "column_value": ["2024-01-10", "2024-01-10", "2024-01-10", "2024-01-10"]
                        }},
                        {{
                            "column_name": "shift_type",
                            "column_value": ["attend_roster", "leave_roster", "attend_roster", "leave_roster"]
                        }}
                    ]
                }}
                
                SQL Query Results: {sql_result}
                
                Return ONLY the JSON object with no additional text or explanation.

                IF THE SQL QUERY RESULTS ARE EMPTY (e.g. "I don't know"), RETURN A MESSAGE THAT SAYS "I apologize, but I couldn't find the information you're looking for. Could you please rephrase your question?"
                
                DO NOT UNDER ANY CIRCUMSTANCES MAKE UP DATA THAT IS NOT IN THE SQL QUERY RESULTS.

                Make sure when you return the data, that all the column values are the same length.
                """,
                input_variables=["input", "sql_result"]
            ),
            'i': PromptTemplate(
                template="""
                You are MOHR AI, a helpful database assistant.
                
                Original Question: {input}
                
                
                SQL Query Results:
                {sql_result}
                
                IMPORTANT RULES:
                1. Always start with a nice greeting and introduce yourself as MOHR AI
                2. Reference the original question in your response
                3. ONLY use data that appears in the SQL query results above
                4. DO NOT make up or infer any additional data
                5. If the SQL results are empty or null, say so clearly
                6. Format numbers consistently (2 decimal places for currency)
                7. Include the actual query results in your response
                8. If you're not sure about something, say so instead of making assumptions
                9. Structure your response to directly answer the original question
                10. If the question requires a bit of creativity where the answer is not directly data from the database. Generate a response that is relevant to the question.
                
                Respond in a clear, professional manner.
                """,
                input_variables=["input", "sql_result"]
            ),
            'c': PromptTemplate(
                template="""
                You are MOHR AI, a friendly conversational AI.
                
                Original Question: {input}
                
                
                SQL Query Results:
                {sql_result}
                
                IMPORTANT RULES:
                1. Always start with a nice greeting and introduce yourself as MOHR AI
                2. Reference the user's question in your response
                3. Be conversational but ONLY use facts from the SQL results
                4. DO NOT make up or infer data that isn't in the results
                5. Keep responses natural and casual while being accurate
                6. If specific numbers are shown, use them exactly as they appear
                7. Feel free to ask follow-up questions related to the original question
                8. If the results are empty or unclear, say so honestly
                9. Make your response feel like a natural conversation
                
                Respond in a friendly, conversational tone.
                """,
                input_variables=["input", "sql_result"]
            )
        }

        # Initialize database connection
        username = os.getenv("AZURE_SQL_USERNAME")
        password = os.getenv("AZURE_SQL_PASSWORD")
        server_name = os.getenv("AZURE_SQL_SERVER")
        database_name = os.getenv("AZURE_SQL_DATABASE")

        connection_string = (
            f"mssql+pyodbc://{username}:{password}@{server_name}"
            f"/{database_name}"
            "?driver=ODBC+Driver+18+for+SQL+Server"
            "&TrustServerCertificate=yes"
            "&timeout=30"
        )
        
        self.database = SQLDatabase.from_uri(connection_string)
        
        self.structured_llm = self.llm_for_agent.with_structured_output(QueryOutput)
        
        self.prompt = PromptTemplate(
            template='''
            Given an input question, create a syntactically correct {dialect} query to run to help find the answer. \
            Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. \
            You can order the results by a relevant column to return the most interesting examples in the database.

            Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

            Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
            
            IMPORTANT:
            Always add "SET CONTEXT_INFO {tenant_id}" in the top of your query.
            If you are asked to get ANY INFORMATION ABOUT A name or if youre looking for a name, look for the column called "EnglishName" EXCLUSIVELY IN THE EMPLOYEES TABLE.  
            
            The following queries were already tried and returned no results, please try a different approach:
            {failed_queries}

            Only use the following tables:
            {table_info}

            Question: {input}''',
            input_variables=['input', 'table_info', 'dialect', 'top_k', 'tenant_id', 'failed_queries']
        )
        
        # Initialize SchemaManager
        self.schema_manager = SchemaManager(schema_file_path)
        self.schema_manager.parse_schema()
        self.parsed_schema = self.schema_manager.get_table_info()

        
    def get_query(self, state: State):
        """Generate an SQL query based on user input and schema."""
        logger.info(f"Generating query for question: {state['question']}")
        logger.info(f"Using tenant_id: {state['tenant_id']}")
        
        # Safely get failed queries with default empty list
        failed_queries = state.get('failed_queries', [])
        failed_queries_text = "\n".join(failed_queries) if failed_queries else "None"
        
        table_info = json.dumps(self.parsed_schema, indent=4)
        prompt = self.prompt.format(
            input=state['question'],
            table_info=table_info,
            dialect=self.database.dialect,
            top_k=10,
            tenant_id=state['tenant_id'],
            failed_queries=failed_queries_text
        )
        logger.info("Generated prompt for query creation")
        
        result = self.structured_llm.invoke(prompt)
        logger.info(f"Generated SQL query: {result['query']}")
        return {"query": result["query"]}
    

    def get_prompt(self, mode):
        """Retrieve the prompt template for the specified mode.

        Args:
            mode (str): The response mode ('r', 'i', or 'c')

        Returns:
            PromptTemplate: The prompt template for the specified mode

        Raises:
            ValueError: If an invalid mode is provided
        """
        if mode not in self.prompts:
            raise ValueError(f"Invalid mode: {mode}")
        return self.prompts[mode]

    def get_tenant_id(self, username):
        """Retrieve the tenant ID for a given username.

        Args:
            username (str): The username to look up

        Returns:
            str: The tenant ID associated with the username
        """
        tenant_query = f"""
        SELECT TenantID FROM Tenants WHERE Name = '{username}'
        """
        result = self.database.run(tenant_query)
        return ''.join(filter(str.isdigit, result))

    def get_llm(self, mode):
        """Initialize a ChatOpenAI instance with mode-specific temperature.

        Args:
            mode (str): The response mode ('r', 'i', or 'c')

        Returns:
            ChatOpenAI: Configured language model instance
        """
        return ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=self.temperature[mode]
        )

    def get_all_tenants(self):
        """Retrieve a list of all tenant names from the database.

        Returns:
            List[str]: List of tenant names
        """
        tenant_query = f"""
        SELECT Name FROM Tenants
        """
        result = self.database.run(tenant_query)
        # Clean up the result by removing tuples and extra characters
        tenant_list = [name[0] for name in eval(result)]
        return tenant_list


    def invoke_prompt(self, mode, input_text, username):
        """Process user input and generate appropriate responses using SQL and LLM."""
        llm = self.get_llm(mode)
        tenant_id = self.get_tenant_id(username)

        # Initialize state with empty failed_queries list
        state: State = {
            'tenant_id': tenant_id,
            'question': input_text,
            'query': '',
            'result': '',
            'answer': '',
            'failed_queries': []
        }

        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Get the query
                query_result = self.get_query(state)
                query = query_result['query']

                # Validate the query
                if not self.validate_query(query):
                    state['failed_queries'].append(f"Attempt {attempt + 1}: {query} (Invalid query)")
                    continue

                # Update state with validated query
                state['query'] = query

                # Execute query and get results
                try:
                    sql_results = self.get_info_from_sql(query)
                    state['result'] = sql_results

                    # If results are empty or error
                    if not sql_results or "don't know" in sql_results.lower():
                        state['failed_queries'].append(f"Attempt {attempt + 1}: {query} (No results)")
                        continue

                    # If we got results, process them based on mode
                    if mode == 'r':
                        try:
                            logger.debug(f"Raw SQL results: {sql_results[:200]}...")
                            # Create a safe evaluation context with both datetime and date
                            eval_context = {
                                "datetime": datetime,
                                "date": date
                            }
                            data = eval(sql_results, eval_context)
                            logger.debug(f"Evaluated data: {str(data)[:200]}...")
                            if not data:
                                return self.handle_unknown_response(mode, input_text)
                            
                            # Extract column names from the query
                            query_lower = query.lower()
                            select_part = query_lower[query_lower.find('select') + 6:query_lower.find('from')].strip()
                            columns = [col.strip().split()[-1] for col in select_part.split(',')]
                            
                            # Format the data for the response
                            formatted_data = {
                                "table_data": []
                            }
                            
                            # Create arrays for each column
                            for i, col in enumerate(columns):
                                column_data = {
                                    "column_name": col,
                                    "column_value": [row[i] for row in data]
                                }
                                formatted_data["table_data"].append(column_data)
                            
                            return formatted_data

                        except Exception as e:
                            logger.error(f"Error parsing raw results: {e}")
                            state['failed_queries'].append(f"Attempt {attempt + 1}: {query} (Parse error)")
                            continue

                    # For informative or conversational modes
                    prompt = self.get_prompt(mode)
                    chain = prompt | llm | self.parser[mode]
                    
                    # Process sql_results to handle datetime before sending to LLM
                    if isinstance(sql_results, str):
                        try:
                            processed_data = eval(sql_results, {"datetime": datetime})
                            if isinstance(processed_data, (list, tuple)):
                                processed_data = [
                                    tuple(
                                        val.strftime('%Y-%m-%d') if isinstance(val, datetime) else val
                                        for val in row
                                    )
                                    for row in processed_data
                                ]
                            sql_results = str(processed_data)
                        except Exception as e:
                            logger.warning(f"Failed to process datetime in results for LLM: {e}")
                    
                    format_input = {
                        "input": input_text,
                        "sql_result": sql_results
                    }
                    answer = chain.invoke(format_input)
                    state['answer'] = answer
                    return answer

                except Exception as sql_error:
                    logger.error(f"SQL execution error: {sql_error}")
                    state['failed_queries'].append(f"Attempt {attempt + 1}: {query} (SQL error: {str(sql_error)})")
                    continue

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in attempt {attempt + 1}: {error_msg}")
                state['failed_queries'].append(f"Attempt {attempt + 1}: General error: {error_msg}")
                continue

        # If we've exhausted all retries, return a failure response
        logger.warning(f"All {max_retries} attempts failed. Failed queries: {state['failed_queries']}")
        return self.handle_unknown_response(mode, input_text)

    #TODO: change if statements to create another chain using llm chain to validate the query
    def validate_query(self, query: str) -> bool:
        """Validate the SQL query for safety and correctness using schema."""
        logger.info(f"Validating query: {query}")
        query_lower = query.lower()
        
        try:
            # Check for required SET CONTEXT_INFO
            if 'set context_info' not in query_lower:
                logger.warning("Validation failed: Missing SET CONTEXT_INFO")
                return False

            # Check for dangerous keywords
            dangerous_keywords = ['drop', 'delete', 'truncate', 'update', 'insert', 'create', 'alter']
            if any(keyword in query_lower for keyword in dangerous_keywords):
                logger.warning(f"Validation failed: Contains dangerous keyword")
                return False

            # Validate SELECT query
            if 'select' not in query_lower:
                logger.warning("Validation failed: Not a SELECT query")
                return False

            # Validate column names against schema
            for table, columns in self.parsed_schema.items():
                if table.lower() in query_lower:
                    referenced_columns = [col.lower() for col in columns if col.lower() in query_lower]
                    if not referenced_columns:
                        logger.warning(f"Validation failed: No valid columns from {table} found in query")
                        return False

            logger.info("Query validation successful")
            return True

        except Exception as e:
            logger.error(f"Query validation error: {e}")
            return False

    def get_info_from_sql(self, query):
        """Execute a SQL query and return the results."""
        logger.info("Executing SQL query...")
        try:
            result = self.database.run(query)
            logger.info(f"Query executed successfully. Result length: {len(str(result))}")
            
            # Convert the result to handle datetime objects
            if isinstance(result, str):
                try:
                    # Add both datetime and date to evaluation context
                    eval_context = {
                        "datetime": datetime,
                        "date": date
                    }
                    processed_result = eval(result, eval_context)
                    
                    # Convert datetime objects in the results
                    if isinstance(processed_result, (list, tuple)):
                        processed_result = [
                            tuple(
                                val.strftime('%Y-%m-%d') if isinstance(val, (datetime, date)) else val
                                for val in row
                            )
                            for row in processed_result
                        ]
                    
                    return str(processed_result)
                except Exception as e:
                    logger.warning(f"Failed to process datetime in result: {e}")
                    return result
            return result
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            raise
        
    def create_csv(self, mode, output):
        """Process query output and create a pandas DataFrame.

        Args:
            mode (str): Response mode ('r', 'i', or 'c')
            output (Union[Dict, str]): Query output to process

        Returns:
            Tuple[Optional[pd.DataFrame], str]: Tuple containing:
                - DataFrame of processed data (or None if error)
                - Status message or error description
        """
        try:
            if mode == 'r':
                if isinstance(output, dict) and "table_data" in output:
                    # Create a dictionary for pandas DataFrame
                    data_dict = {}
                    
                    # Extract data from the structured format
                    for column in output["table_data"]:
                        col_name = column["column_name"]
                        col_values = column["column_value"]
                        data_dict[col_name] = col_values
                    
                    # Create DataFrame
                    df = pd.DataFrame(data_dict)
                    
                    # Clean up DataFrame
                    df = df.replace('None', pd.NA)
                    
                    # Sort if there's a name column
                    name_cols = [col for col in df.columns if 'name' in col.lower()]
                    if name_cols:
                        df = df.sort_values(name_cols[0])
                    df = df.reset_index(drop=True)
                    
                    # Create summary
                    num_rows = len(df)
                    summary = f"Found {num_rows} records in the data."
                    
                    return df, summary
                else:
                    logger.warning("Invalid output format for CSV creation")
                    return None, "Invalid data format"
                    
            return None, "Mode not supported for CSV creation"
                
        except Exception as e:
            logger.error(f"Error in create_csv: {e}")
            return None, f"Error processing data: {str(e)}"
        
    def handle_unknown_response(self, mode, input_text):
        """Handle cases where the SQL agent doesn't know the answer."""
        llm = self.get_llm(mode)
        prompt = self.get_prompt(mode)
        chain = prompt | llm | self.parser[mode]
        format_input = {
            "input": input_text,
            "sql_result": "I don't have specific data from the database for this query, but I'll try to help you with what I know."
        }
        return chain.invoke(format_input)

if __name__ == "__main__":
    db = DatabaseBackend()
    print(sorted(db.get_all_tenants()))