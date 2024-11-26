from dotenv import load_dotenv
import pyodbc
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Any, TypedDict
import json
import os
import pandas as pd

load_dotenv()


class SQLResponse(TypedDict):
    sql_query: str
    column_names: List[str]

class DatabaseBackend:
    def __init__(self):
        # Initialize temperature
        self.temperature = {
            'r': 0,
            'i': 0.4,
            'c': 0.9
        }

        self.llm_for_agent = ChatOpenAI(
            model_name="gpt-4o-2024-11-20", 
            temperature=0
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
        
        self.SQLAgent = create_sql_agent(
            llm=self.llm_for_agent, 
            db=self.database, 
            verbose=True,
            agent_executor_kwargs={
                "handle_parsing_errors": True,
                "input_format_instructions": """Do not use markdown formatting in your queries.
                    When using sql_db_query, provide the raw SQL without any backticks or ```sql tags."""
            }
        )

    def get_prompt(self, mode):
        if mode not in self.prompts:
            raise ValueError(f"Invalid mode: {mode}")
        return self.prompts[mode]

    def get_tenant_id(self, username):
        tenant_query = f"""
        SELECT TenantID FROM Tenants WHERE Name = '{username}'
        """
        result = self.database.run(tenant_query)
        return ''.join(filter(str.isdigit, result))

    def get_llm(self, mode):
        return ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=self.temperature[mode]
        )

    def get_all_tenants(self):
        tenant_query = f"""
        SELECT Name FROM Tenants
        """
        result = self.database.run(tenant_query)
        # Clean up the result by removing tuples and extra characters
        tenant_list = [name[0] for name in eval(result)]
        return tenant_list


    def invoke_prompt(self, mode, input_text, username):
        llm = self.get_llm(mode)
        tenant_id = self.get_tenant_id(username)
        
        # Define JSON parser
        parser = JsonOutputParser(pydantic_object=SQLResponse)

        try:
            full_input = f"""You are a SQL query assistant. Follow these rules precisely:
            
                REQUIRED FOR EVERY QUERY:
                1. Start with: SET CONTEXT_INFO {tenant_id}
                2. Return complete results (no LIMIT/TOP unless specified)
                3. For names, always use EnglishName column
                4. Include EnglishName when returning employee-related data
                5. Make column names user-friendly (e.g., "Employee ID" instead of "Id")
                6. You can explore tables and check schemas as needed
                7. When you are excuting the queries, make sure to NOT include the ```sql tags as it results in this error: 
                ```Error: (pyodbc.ProgrammingError) ('42000', "[42000] [Microsoft][ODBC Driver 18 for SQL Server][SQL Server]Incorrect syntax near '`'. (102) (SQLExecDirectW)")
                IF YOU GET THIS ERROR, REMOVE THE ```sql TAGS FROM YOUR QUERY.
                8. Once you have determined the correct query and tested it successfully and got the results, SAY I NOW KNOW THE FINAL ANSWER AND STOP PROCESSING AND RETURN A JSON OBJECT WITH EXACTLY THIS STRUCTURE:
                   ```json
                   {{
                       "sql_query": "SET CONTEXT_INFO {tenant_id}; YOUR_FINAL_SQL_QUERY",
                       "column_names": ["Your", "Column", "Names"]
                   }}
                   ```
                9. make sure to return the JSON object with the ```json tags.
                Question: {input_text}"""

            # Get the response from the agent
            agent_response = self.SQLAgent.invoke(full_input)
            
            # Parse the JSON response from the output
            output = agent_response.get('output', '')
            response_dict = parser.parse(output)
            
            sql_query = response_dict['sql_query']
            column_names = response_dict['column_names']
            
            print(f"\nSQL Query: {sql_query}")
            print(f"\nColumn Names: {column_names}")

            # Execute the SQL query using get_info_from_sql
            sql_results = self.get_info_from_sql(sql_query)
            
            # For 'r' mode, return both results and column names
            if mode == 'r':
                return {
                    'results': sql_results,
                    'column_names': column_names  # Already a list from JSON parsing
                }
            
            # For other modes, continue with existing LLM formatting
            prompt = self.get_prompt(mode)
            chain = prompt | llm | self.parser[mode]
            format_input = {
                "input": input_text,
                "sql_result": sql_results
            }
            llm_response = chain.invoke(format_input)
            return llm_response

        except Exception as e:
            error_msg = str(e)
            print(f"Error in invoke_prompt: {error_msg}")
            return f"Error executing query: {error_msg}"
        
    def get_info_from_sql(self, query):
        result = self.database.run(query)
        return result
        
    def create_csv(self, mode, output):
        try:
            if mode == 'r':
                print("\n=== Debug Create CSV ===")
                print(f"Input type: {type(output)}")
                
                if isinstance(output, dict):
                    print("\nDictionary contents:")
                    for key, value in output.items():
                        print(f"\nKey: {key}")
                        print(f"Value type: {type(value)}")
                        print(f"Value: {value[:500] if isinstance(value, str) else value}")
                    
                    try:
                        # Extract results and column names
                        data_list = eval(output['results'])
                        columns = output['column_names']  # Already a list from invoke_prompt
                        
                        print("\nAfter processing:")
                        print(f"Data list type: {type(data_list)}")
                        print(f"First row: {data_list[0] if data_list else 'No data'}")
                        print(f"Columns: {columns}")
                        
                        if not data_list:
                            return None, "No data found"
                        
                        # Initialize dictionary with empty lists for each column
                        table_dict = {col: [] for col in columns}
                        
                        # Populate data
                        for row in data_list:
                            for col_name, value in zip(columns, row):
                                table_dict[col_name].append(value)
                        
                        # Create DataFrame
                        df = pd.DataFrame(table_dict)
                        print(f"\nCreated DataFrame with shape: {df.shape}")
                        
                        # Clean up DataFrame
                        df = df.replace('None', pd.NA)
                        
                        # If any column contains 'name', strip whitespace
                        name_cols = [col for col in df.columns if isinstance(col, str) and 'name' in col.lower()]
                        for col in name_cols:
                            df[col] = df[col].str.strip()
                        
                        # Sort by first name column if it exists
                        if name_cols:
                            df = df.sort_values(name_cols[0])
                        df = df.reset_index(drop=True)
                        
                        # Create summary message
                        num_rows = len(df)
                        summary = f"Found {num_rows} records in the data."
                        print(f"\nSummary: {summary}")
                        
                        return df, summary
                        
                    except Exception as e:
                        print(f"\nError processing dictionary: {e}")
                        print(f"Error type: {type(e)}")
                        return None, f"Error processing data: {str(e)}"
                else:
                    print("Input is not a dictionary")
                    return None, "Invalid input type"
                    
            return output, "Data processed successfully"
                
        except Exception as e:
            print(f"Error in create_csv: {e}")
            return None, f"Error processing data: {str(e)}"
        

if __name__ == "__main__":
    db = DatabaseBackend()
    print(db.get_all_tenants())