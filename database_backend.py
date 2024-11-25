from dotenv import load_dotenv
import pyodbc
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Any
import json
import os
import pandas as pd

load_dotenv()

class Column_in_Table(BaseModel):
    column_name: str = Field(examples=["column1", "column2"], description="The name of the column")
    column_value: Any = Field(examples=[1, "value", 3.14], description="The value of the column")

class Table(BaseModel):
    table_data: List[Column_in_Table] = Field(examples=[
        {"column_name": "column1", "column_value": "value1"}, 
        {"column_name": "column2", "column_value": "value2"}],
        description="A list of columns in the table")

class DatabaseBackend:
    def __init__(self):
        # Initialize temperature
        self.temperature = {
            'r': 0,
            'i': 0.4,
            'c': 0.9
        }

        self.llm_for_agent = ChatOpenAI(
            model_name="gpt-4o", 
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
            agent_executor_kwargs = {"handle_parsing_errors": True}
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

        try:
            full_input = f"""You are a SQL query assistant. Follow these rules precisely:
            
                REQUIRED FOR EVERY QUERY:
                1. Start with: SET CONTEXT_INFO {tenant_id}
                2. Return complete results (no LIMIT/TOP unless specified)
                3. For names, always use EnglishName column
                4. Include EnglishName when returning employee-related data
                5. When returning any sql query, return the raw sql query, not the markdown formatting or explanations

                
                OUTPUT FORMAT:
                1. Results must be a list of tuples:
                   - First tuple: Exact column names from database
                   - Following tuples: The data rows
                2. After query execution, write "Final Answer:" followed by complete, untruncated results
                3. If the query returns no results, write "Final Answer: I apologize, but I couldn't find the information you're looking for. Could you please rephrase your question?"
                4. DO NOT TRUNCATE THE RESULTS, RETURN THE ENTIRE RESULT SET.
                5. If there's an empty cell, write "None" instead of an empty string for that cell only.

                EXAMPLE OUTPUT:
                Final Answer: [('EnglishName', 'Department'), ('John Smith', 'IT'), ('Jane Doe', 'HR')]

                Question: {input_text}"""

            # This returns an AgentFinish object
            agent_response = self.SQLAgent.invoke(full_input)
            
            # Let's add some debugging to see the full response
            print("Full agent response:", agent_response)
            
            # Get the output, which is typically everything after "Final Answer:"
            final_answer = agent_response.get('output', '')
            print("\n\nFinal answer:", final_answer)
            print("\n\nFinal answer type:", type(final_answer))
            
            print("\n=== Debug SQL Agent Response ===")
            print(f"Response type: {type(final_answer)}")
            print(f"First 500 chars: {final_answer[:500]}")
            print("=== End Debug ===\n")
            
            if not final_answer or "I don't know" in final_answer:
                return "I apologize, but I couldn't find the information you're looking for. Could you please rephrase your question?"

            # For 'r' mode, return the SQL results directly
            if mode == 'r':
                # Extract just the data portion (everything after "Final Answer:")
                if "Final Answer:" in final_answer:
                    data_portion = final_answer.split("Final Answer:")[1].strip()
                    # Add print statement for debugging
                    print("Data portion:", data_portion)
                    return data_portion
                # Add print statement for debugging
                print("Final answer (no Final Answer: found):", final_answer)
                return final_answer
            
            # For other modes, continue with existing LLM formatting
            prompt = self.get_prompt(mode)
            chain = prompt | llm | self.parser[mode]
            format_input = {
                "input": input_text,
                "sql_result": final_answer
            }
            llm_response = chain.invoke(format_input)
            # Add print statement for debugging
            print("LLM Response:", llm_response)
            return llm_response
        
        except Exception as e:
            error_msg = str(e)
            print(f"Error in invoke_prompt: {error_msg}")  # Add error logging
            if "OUTPUT_PARSING_FAILURE" in error_msg:
                return final_answer
            return f"Error executing query: {error_msg}"
        
    def create_csv(self, mode, output):
        try:
            if mode == 'r':
                print("\n=== Debug Create CSV ===")
                print(f"Input type: {type(output)}")
                
                if isinstance(output, str):
                    try:
                        # Convert string to list of tuples
                        data_list = eval(output)
                        print(f"\nConverted to list. Length: {len(data_list)}")
                        print(f"First tuple: {data_list[0]}")
                        
                        if not data_list:
                            return None, "No data found"
                        
                        # Get column names from first tuple
                        columns = list(data_list[0])
                        num_columns = len(columns)
                        print(f"\nDetected {num_columns} columns: {columns}")
                        
                        # Initialize dictionary with empty lists for each column
                        table_dict = {col: [] for col in columns}
                        
                        # Populate data
                        for row in data_list[1:]:  # Skip header tuple
                            if len(row) != num_columns:
                                print(f"\nWarning: Row has different length than headers")
                                print(f"Expected {num_columns}, got {len(row)}")
                                print(f"Row: {row}")
                                continue
                            
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
                        print(f"Summary: {summary}")
                        print("=== End Debug ===\n")
                        
                        return df, summary
                        
                    except Exception as e:
                        print(f"Error processing list of tuples: {e}")
                        print(f"First 500 chars of input: {output[:500]}")
                        return None, f"Error processing data: {str(e)}"
                else:
                    print("Input is not a string")
                    return None, "Invalid input type"
                    
            return output, "Data processed successfully"
                
        except Exception as e:
            print(f"Error in create_csv: {e}")
            print(f"Error type: {type(e)}")
            print(f"Error args: {e.args}")
            return None, f"Error processing data: {str(e)}"
        

if __name__ == "__main__":
    db = DatabaseBackend()
    print(db.get_all_tenants())