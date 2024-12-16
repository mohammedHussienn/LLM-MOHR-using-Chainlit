from dotenv import load_dotenv
import pyodbc
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing import List, Any, TypedDict, Optional
import os
import pandas as pd
from typing_extensions import Annotated
import logging
import json
from datetime import datetime, date


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


load_dotenv()

class State(TypedDict):
    """State TypedDict with required fields."""
    tenant_id: int
    question: str
    query: str
    valid: bool
    result: str
    answer: str
    failed_queries: List[str]

class QueryChain:
    """Chain responsible for generating SQL queries."""
    def __init__(self, schema_file_path: str, database, llm=None):
        # Read schema directly from file
        with open(schema_file_path, 'r') as f:
            self.schema = f.read()
            
        self.database = database
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.output_parser = StrOutputParser()
        
        self.prompt = PromptTemplate(
            template='''
            Given an input question, create a syntactically correct {dialect} query to run to help find the answer. \
            Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results using "TOP" syntax. \
            
            Never query for all the columns from a specific table, only ask for the few relevant columns given the question.

            Pay attention to use only the column names you can see in the schema description. \
            Also, pay attention to which column is in which table.
            
            IMPORTANT RULES:
            1. Always add "SET CONTEXT_INFO {tenant_id}" at the top of your query
            2. Do not use Top N in your query unless the user specifically asks for a specific number of results.
            3. If you're asked to get ANY INFORMATION ABOUT A name or if you're looking for a name, look for the column called "EnglishName" EXCLUSIVELY IN THE EMPLOYEES TABLE
            
            The following queries were already tried and returned no results, please try a different approach:
            {failed_queries}

            Schema Description:
            {schema}

            Question: {input}
            
            Return ONLY the SQL query with no additional text or explanation.
            ''',
            input_variables=['input', 'schema', 'dialect', 'top_k', 'tenant_id', 'failed_queries']
        )

    def generate_query(self, question: str, tenant_id: int, failed_queries: List[str] = []) -> str:
        """Generate SQL query from user question."""
        failed_queries_text = "\n".join(failed_queries) if failed_queries else "None"
        
        # Add caching to reduce API calls
        cache_key = f"{question}_{tenant_id}_{failed_queries_text}"
        if hasattr(self, '_query_cache') and cache_key in self._query_cache:
            logger.info(f"Using cached query: {self._query_cache[cache_key]}")
            return self._query_cache[cache_key]
        
        chain = self.prompt | self.llm | self.output_parser
        query = chain.invoke({
            "input": question,
            "schema": self.schema,
            "dialect": self.database.dialect,
            "top_k": 10,
            "tenant_id": tenant_id,
            "failed_queries": failed_queries_text
        })
        
        query = query.strip()
        logger.info(f"Generated SQL query: {query}")
        
        if not hasattr(self, '_query_cache'):
            self._query_cache = {}
        self._query_cache[cache_key] = query
        return query

class QueryValidator:
    """Chain responsible for validating SQL queries."""
    def __init__(self, schema_file_path: str, database, llm=None):
        # Read schema directly from file
        with open(schema_file_path, 'r') as f:
            self.schema = f.read()
            
        self.database = database
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        
        self.prompt = PromptTemplate(
            template='''
            Validate the following SQL query against these requirements:
            
            1. Must contain "SET CONTEXT_INFO"
            2. Must be a SELECT query only
            3. Must not contain dangerous keywords (DROP, DELETE, TRUNCATE, UPDATE, INSERT, CREATE, ALTER)
            4. Must reference valid columns from the schema
            
            Schema:
            {schema}
            
            Query to validate:
            {query}
            
            Return ONLY "VALID" if the query is valid, or explain why it's invalid starting with "INVALID:".
            ''',
            input_variables=['schema', 'query']
        )
        
    def validate(self, query: str) -> tuple[bool, str]:
        """Validate query and return boolean result and reason."""
        chain = self.prompt | self.llm | StrOutputParser()
        
        validation_result = chain.invoke({
            "schema": self.schema,
            "query": query
        }).strip()
        
        is_valid = "VALID" in validation_result.upper()
        reason = "" if is_valid else validation_result
        
        return is_valid, reason

    def execute_if_valid(self, query: str) -> Optional[str]:
        """Validate and execute query if valid."""
        is_valid, reason = self.validate(query)
        if is_valid:
            try:
                clean_query = query.replace('```sql', '').replace('```', '').strip()
                logger.info(f"Executing SQL query: {clean_query}")
                result = self.database.run(clean_query)
                logger.info(f"Query result: {result}")
                return str(result)
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                return None
        else:
            logger.warning(f"Query validation failed: {reason}")
            return None

def create_query_output(query: str) -> dict:
    """Create a query output dictionary."""
    return {"query": query}

class DatabaseBackend:
    """Handles database operations and LLM interactions for SQL queries.

    This class manages database connections, SQL queries, and language model interactions
    for different response modes (raw, informative, conversational).
    """

    def __init__(self, schema_file_path="newSchema.txt"):
        # Initialize temperature
        self.temperature = {
            'r': 0,
            'i': 0.4,
            'c': 0.9
        }

        # Initialize OpenAI client with rate limiting
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_retries=3,
            max_tokens=1000
        )

        # Initialize parsers
        self.parser = {
            'r': StrOutputParser(),
            'i': StrOutputParser(),
            'c': StrOutputParser()
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
        
        # Initialize chains with schema file path
        self.query_chain = QueryChain(schema_file_path, self.database, llm=self.llm)
        self.validator = QueryValidator(schema_file_path, self.database, llm=self.llm)
        
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
                7. Use null for missing values (not None, undefined, or empty string)
                
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
        """Retrieve the tenant ID for a given username."""
        tenant_query = f"""
        SELECT TenantID FROM Tenants WHERE Name = '{username}'
        """
        result = self.database.run(tenant_query)
        # Convert result to string and extract only digits
        result_str = str(result)
        digits = ''.join(char for char in result_str if char.isdigit())
        return int(digits) if digits else None

    def get_llm(self, mode):
        """Initialize a ChatOpenAI instance with mode-specific temperature.

        Args:
            mode (str): The response mode ('r', 'i', or 'c')

        Returns:
            ChatOpenAI: Configured language model instance
        """
        return ChatOpenAI(
            model="gpt-4o-mini", 
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
        tenant_list = [name[0] for name in eval(str(result))]
        return tenant_list

    def get_query(self, state: State) -> str:
        """Generate SQL query based on user input and schema."""
        return self.query_chain.generate_query(
            question=state['question'],
            tenant_id=int(state['tenant_id']),
            failed_queries=state.get('failed_queries', [])
        )

    def validate_query(self, query: str) -> tuple[bool, str]:
        """Validate the SQL query."""
        return self.validator.validate(query)

    def create_csv(self, mode: str, output: Any) -> tuple[Optional[pd.DataFrame], str]:
        """Process query output and create a pandas DataFrame."""
        if mode == 'r':
            logger.info(f"Raw output received: {output}")
            
            try:
                # Handle string input
                if isinstance(output, str):
                    output = json.loads(output)
                
                if isinstance(output, dict) and "table_data" in output:
                    # Create a dictionary for pandas DataFrame
                    data_dict = {}
                    
                    # Extract data from the structured format
                    for column in output["table_data"]:
                        col_name = column["column_name"]
                        # Convert None values to pandas NA
                        col_values = [pd.NA if v is None else v for v in column["column_value"]]
                        data_dict[col_name] = col_values
                    
                    # Create DataFrame
                    df = pd.DataFrame(data_dict)
                    logger.info(f"Created DataFrame with shape: {df.shape}")
                    
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
                    logger.warning("Missing 'table_data' in output")
                    return None, "Invalid data format: missing table_data"
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                return None, f"Invalid JSON format: {str(e)}"
            except Exception as e:
                logger.error(f"Error creating DataFrame: {str(e)}")
                return None, f"Error processing data: {str(e)}"
                
        return None, "Mode not supported for CSV creation"

    def invoke_prompt(self, mode: str, input_text: str, username: str) -> str:
        """Process user input and return formatted response."""
        tenant_id = self.get_tenant_id(username)
        if tenant_id is None:
            return "Could not find tenant ID. Please check the tenant name."
        
        result = self.validator.execute_if_valid(self.query_chain.generate_query(
            input_text, tenant_id, []
        ))
        
        if not result:
            return "I couldn't find any relevant information. Please try rephrasing your question."
        
        llm = self.get_llm(mode)
        prompt = self.get_prompt(mode)
        chain = prompt | llm | self.parser[mode]
        
        return chain.invoke({
            "input": input_text,
            "sql_result": result
        })

if __name__ == "__main__":
    db = DatabaseBackend()
    print(sorted(db.get_all_tenants()))