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
from tiktoken import encoding_for_model
import functools
from collections import defaultdict
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global token counter
token_counts = defaultdict(lambda: {'input': 0, 'output': 0, 'calls': 0})

def count_tokens(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        enc = encoding_for_model("gpt-4o")
        
        if len(args) > 1 and isinstance(args[1], dict):
            state = args[1]
            
            # Calculate full prompt input tokens based on class type
            if isinstance(self, QueryGeneration):
                full_prompt = self.prompt.format(
                    input=state['question'],
                    schema=self.schema,
                    dialect=self.database.dialect,
                    top_k=10,
                    tenant_id=state['tenant_id'],
                    failed_queries="\n".join(state['failed_queries']) if state['failed_queries'] else "None"
                )
                input_tokens = len(enc.encode(full_prompt))
            
            elif isinstance(self, QueryValidation):
                full_prompt = self.prompt.format(
                    schema=self.schema,
                    query=state['query']
                )
                input_tokens = len(enc.encode(full_prompt))
            
            else:
                # For other classes, keep existing logic
                input_text = f"{state.get('question', '')} {state.get('query', '')}"
                input_tokens = len(enc.encode(input_text))
            
            result = func(*args, **kwargs)
            
            if isinstance(result, dict):
                # Calculate output tokens based on class type
                if isinstance(self, QueryGeneration):
                    output_text = json.dumps({
                        "query": result.get('query', ''),
                        "column_names": result.get('column_names', [])
                    })
                    output_tokens = len(enc.encode(output_text))
                
                elif isinstance(self, QueryValidation):
                    output_text = "VALID" if result.get('valid') else f"INVALID: {result.get('answer', '')}"
                    output_tokens = len(enc.encode(output_text))
                
                else:
                    # For other modes (like raw mode)
                    if state.get('mode') == 'r':
                        output_text = f"Found {result.get('answer', '')}"
                    else:
                        output_text = f"{result.get('answer', '')} {result.get('query', '')}"
                    output_tokens = len(enc.encode(output_text))
                
                func_name = f"{self.__class__.__name__}.{func.__name__}"
                token_counts[func_name]['input'] += input_tokens
                token_counts[func_name]['output'] += output_tokens
                token_counts[func_name]['calls'] += 1
                
                logger.info(f"Function {func_name}: Input tokens: {input_tokens}, Output tokens: {output_tokens}")
            
            return result
        return func(*args, **kwargs)
    return wrapper

def print_token_report():
    """Print a summary report of token usage."""
    print("\n" + "="*50)
    print("TOKEN USAGE REPORT")
    print("="*50)
    print(f"{'Function Name':<30} {'Calls':<8} {'Input':<10} {'Output':<10} {'Total':<10}")
    print("-"*70)
    
    total_input = 0
    total_output = 0
    total_calls = 0
    
    # Sort the functions by total token usage
    sorted_funcs = sorted(
        token_counts.items(),
        key=lambda x: x[1]['input'] + x[1]['output'],
        reverse=True
    )
    
    for func_name, counts in sorted_funcs:
        calls = counts['calls']
        input_tokens = counts['input']
        output_tokens = counts['output']
        total = input_tokens + output_tokens
        
        total_input += input_tokens
        total_output += output_tokens
        total_calls += calls
        
        print(f"{func_name:<30} {calls:<8} {input_tokens:<10} {output_tokens:<10} {total:<10}")
    
    print("-"*70)
    print(f"{'TOTAL':<30} {total_calls:<8} {total_input:<10} {total_output:<10} {total_input + total_output:<10}")
    print("="*70 + "\n")


load_dotenv()

class State(TypedDict):
    """State TypedDict with required fields."""
    mode: str
    tenant_id: Optional[int]
    question: str
    query: str
    column_names: List[str]
    valid: bool
    result: str
    answer: str
    failed_queries: List[str]
    current_df: Optional[pd.DataFrame]

class ProcessInput:
    """Processes initial user input and creates/updates state."""
    def __init__(self, tenant_id_lookup_fn):
        """Initialize with a function to lookup tenant IDs."""
        self.get_tenant_id = tenant_id_lookup_fn

    def process(self, state: State) -> State:
        """Process the input and update state."""
        if not state['tenant_id']:
            return {**state, 'answer': "Could not find tenant ID. Please check the tenant name."}
        return state

class QueryGeneration:
    """Handles query generation and updates state."""
    def __init__(self, schema_file_path: str, database, llm=None):
        # Initialize with all necessary components
        with open(schema_file_path, 'r') as f:
            self.schema = f.read()
        self.database = database
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.output_parser = JsonOutputParser()
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
            4. VERY IMPORTANT: The column_names array MUST EXACTLY MATCH the columns in your SELECT statement, in the same order.
            
            The following queries were already tried and returned no results, please try a different approach:
            {failed_queries}

            Schema Description:
            {schema}

            Question: {input}
            
            Return a JSON object in the following format:
            {{
                "query": "<your SQL query here>",
                "column_names": ["col1", "col2", "col3"]  // MUST match SELECT columns exactly BUT if they have the same name, you can use aliases for them 
            }}
            
            Ensure the response is valid JSON and that column_names matches your SELECT statement exactly.
            ''',
            input_variables=['input', 'schema', 'dialect', 'top_k', 'tenant_id', 'failed_queries']
        )
    
    @count_tokens
    def process(self, state: State) -> State:
        """Generate query and update state."""
        chain = self.prompt | self.llm | self.output_parser
        response_dict = chain.invoke({
            "input": state['question'],
            "schema": self.schema,
            "dialect": self.database.dialect,
            "top_k": 10,
            "tenant_id": state['tenant_id'],
            "failed_queries": "\n".join(state['failed_queries']) if state['failed_queries'] else "None"
        })
        
        # Extract columns from the query
        query = response_dict['query']
        try:
            # Find the columns in the SELECT statement
            select_part = query.upper().split('FROM')[0].split('SELECT')[1].strip()
            # Handle cases with SET CONTEXT_INFO before SELECT
            if 'SET CONTEXT_INFO' in select_part:
                select_part = select_part.split('SELECT')[1].strip()
            
            # Improved column parsing
            actual_columns = []
            current_column = []
            in_parentheses = 0
            
            for char in select_part:
                if char == '(':
                    in_parentheses += 1
                elif char == ')':
                    in_parentheses -= 1
                elif char == ',' and in_parentheses == 0:
                    actual_columns.append(''.join(current_column).strip())
                    current_column = []
                    continue
                current_column.append(char)
            
            # Add the last column
            if current_column:
                actual_columns.append(''.join(current_column).strip())
            
            # Clean up column names and handle aliases
            actual_columns = [col.split(' AS ')[-1].strip() for col in actual_columns]
            
            # Compare with provided column names
            if len(actual_columns) != len(response_dict['column_names']):
                logger.warning(f"Column count mismatch! Query has {len(actual_columns)} columns but names list has {len(response_dict['column_names'])} columns")
                logger.warning(f"Query columns: {actual_columns}")
                logger.warning(f"Provided names: {response_dict['column_names']}")
                # Use the actual columns from the query instead
                response_dict['column_names'] = actual_columns
        except Exception as e:
            logger.error(f"Error parsing query columns: {e}")
            # If we can't parse the query, keep the original column names
            pass
        
        return {
            **state, 
            'query': response_dict['query'],
            'column_names': response_dict['column_names']
        }

class QueryValidation:
    """Handles query validation and updates state."""
    def __init__(self, schema_file_path: str, llm=None):
        with open(schema_file_path, 'r') as f:
            self.schema = f.read()
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
    
    @count_tokens
    def process(self, state: State) -> State:
        """Validate query and update state."""
        chain = self.prompt | self.llm | StrOutputParser()
        validation_result = chain.invoke({
            "schema": self.schema,
            "query": state['query']
        }).strip()
        
        is_valid = "VALID" in validation_result.upper()
        if not is_valid:
            return {
                **state,
                'valid': False,
                'failed_queries': state['failed_queries'] + [state['query']],
                'query': ''
            }
        return {**state, 'valid': True}

class QueryExecution:
    """Handles query execution and updates state."""
    def __init__(self, database):
        self.database = database
    
    def process(self, state: State) -> State:
        """Execute query and update state."""
        if not state['valid']:
            return state
            
        try:
            clean_query = state['query'].replace('```sql', '').replace('```', '').strip()
            result = self.database.run(clean_query)
            
            # Check if result is empty
            result_str = str(result)
            if result_str == '[]' or not result_str:
                # Add current query to failed queries and invalidate
                return {
                    **state,
                    'result': '',
                    'valid': False,
                    'failed_queries': state['failed_queries'] + [clean_query],
                    'query': ''
                }
                
            return {**state, 'result': result_str}
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return {
                **state,
                'result': '',
                'valid': False,
                'failed_queries': state['failed_queries'] + [clean_query],
                'query': ''
            }

class AnswerGeneration:
    """Generates final answer based on mode and updates state."""
    def __init__(self, database):
        self.database = database
    
    @count_tokens
    def process(self, state: State) -> State:
        """Generate answer and update state."""
        if not state['result']:
            return {**state, 'answer': "I couldn't find any relevant information. Please try rephrasing your question."}
        
        try:
            df, message = self.database.create_csv('r', state['result'], state['column_names'])
            if df is not None:
                return {**state, 'answer': message}
            return {**state, 'answer': "Failed to process the data"}
        except Exception as e:
            logger.error(f"Error in raw mode: {e}")
            return {**state, 'answer': f"Error processing data: {str(e)}"}

class DataFrameAgent:
    """Handles pandas DataFrame analysis using LangChain agents."""
    
    def __init__(self, llm=None):
        """Initialize with optional LLM."""
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4o-mini")
        self._agent = None
        
    def _create_agent(self, df: pd.DataFrame):
        """Create pandas DataFrame agent."""
        if self._agent is None:
            self._agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=True,
                allow_dangerous_code=True
            )
        return self._agent
    
    @count_tokens
    def process(self, state: State) -> State:
        """Process a question about the current DataFrame."""
        try:
            df = state.get('current_df')
            if df is None or df.empty:
                return {
                    **state,
                    'answer': "No data available to analyze. Please query for data first.",
                    'mode': 'r'  # Switch back to raw mode if no data
                }
            
            # Create or get pandas DataFrame agent
            agent = self._create_agent(df)
            
            # Run the agent with the user's question
            response = agent.run(state['question'])
            
            return {
                **state,
                'answer': response,
                'mode': 'a'  # Stay in analysis mode
            }
            
        except Exception as e:
            logger.error(f"Error in DataFrame analysis: {e}")
            return {
                **state,
                'answer': f"Error analyzing data: {str(e)}",
                'mode': 'a'  # Stay in analysis mode even on error
            }

class DatabaseBackend:
    """Handles database operations and LLM interactions for SQL queries."""
    def __init__(self, schema_file_path: str):
        self.schema_file_path = schema_file_path
        
        # Initialize OpenAI client
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_retries=3,
            max_tokens=1000
        )

        # Initialize parsers
        self.parser = StrOutputParser()
        
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

    def get_tenant_id(self, username):
        """Get tenant ID for username."""
        tenant_query = f"""
        SELECT TenantID FROM Tenants WHERE Name = '{username}'
        """
        result = self.database.run(tenant_query)
        result_str = str(result)
        digits = ''.join(char for char in result_str if char.isdigit())
        return int(digits) if digits else None

    def get_all_tenants(self):
        """Get list of all tenant names."""
        tenant_query = "SELECT Name FROM Tenants"
        result = self.database.run(tenant_query)
        return [name[0] for name in eval(str(result))]

    def create_csv(self, mode: str, result: str, column_names: List[str]) -> tuple[Optional[pd.DataFrame], str]:
        """Create DataFrame from query output."""
        logger.debug("="*50)
        logger.debug("CREATE_CSV FUNCTION CALLED")
        logger.debug(f"Creating CSV in mode: {mode}")
        logger.debug(f"Column names received: {column_names}")
        logger.debug(f"Raw result length: {len(result)}")
        logger.debug("="*50)
        
        if mode == 'r':
            try:
                # Clean and prepare the result string
                result = result.replace('datetime.datetime', 'datetime')
                result = result.replace('datetime.date', 'date')
                
                # Convert string result to list of tuples
                result_data = eval(result)
                logger.debug(f"Successfully evaluated result data with {len(result_data)} records")
                
                # Create DataFrame with provided column names
                df = pd.DataFrame(result_data, columns=column_names)
                logger.debug(f"Created DataFrame with shape: {df.shape}")
                
                # Handle datetime columns generically
                for col in df.columns:
                    if df[col].dtype == 'object' and df[col].notna().any():
                        first_value = df[col].iloc[0]
                        if isinstance(first_value, (datetime, date)):
                            df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M')
                
                logger.info(f"Successfully created DataFrame with {len(df)} records")
                return df, f"Found {len(df)} records"
                
            except Exception as e:
                logger.error(f"Error creating DataFrame: {str(e)}", exc_info=True)
                return None, f"Error processing data: {str(e)}"
                
        logger.warning(f"Unsupported mode: {mode}")
        return None, "Mode not supported for CSV creation"
