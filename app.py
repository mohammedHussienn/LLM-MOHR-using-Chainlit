import chainlit as cl
from chainlit.input_widget import Select
from database_backend import DatabaseBackend, State, ProcessInput, QueryGeneration, QueryValidation, QueryExecution, AnswerGeneration, print_token_report
import logging
import pandas as pd
from typing import Optional
import json
from datetime import datetime
import io
import contextlib

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from tabulate import tabulate
    TABULATE_INSTALLED = True
except ImportError:
    TABULATE_INSTALLED = False
    logger.warning("tabulate not installed. Tables will be displayed in simple format.")

# Initialize database backend
db = DatabaseBackend(schema_file_path="newSchema.txt")

async def send_data_response(df: pd.DataFrame, summary: str, query: Optional[str] = None):
    """Send formatted data response to chat."""
    if len(df) > 0:
        try:
            # Send the summary and query first
            message = f"{summary}\n\n"
            if query:
                message += f"��� Query used:\n```sql\n{query}\n```"
            
            await cl.Message(content=message).send()

            # Create Excel file in memory
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            
            file_name = f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            await cl.Message(
                content=f"📥 Download complete dataset ({len(df)} records):",
                elements=[
                    cl.File(
                        name=file_name,
                        content=excel_buffer.getvalue(),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                ]
            ).send()
            
        except Exception as e:
            logger.error(f"Error sending data response: {str(e)}", exc_info=True)
            await cl.Message(content=f"Error displaying data: {str(e)}").send()
    else:
        await cl.Message(content="No records found.").send()

@cl.on_chat_start
async def start():
    """Initialize chat session."""
    await cl.Message(
        content="""👋 Welcome to MOHR AI Assistant!

I'm here to help you with your queries. Here's what you can do:
- Ask questions about data
- Get information in different formats
- Have natural conversations about the data

Please select your preferences below to get started!
""").send()
    
    # Setup chat settings
    tenants = sorted(db.get_all_tenants())
    settings = await cl.ChatSettings([
        Select(
            id="mode",
            label="Chat Mode",
            values=["Raw Mode", "Informative Mode", "Conversational Mode"],
            initial_value="Informative Mode"
        ),
        Select(
            id="tenant",
            label="Select Tenant",
            values=tenants,
            initial_value="testmohr"
        )
    ]).send()
    
    # Set default session values
    cl.user_session.set("mode", "i")
    cl.user_session.set("tenant", "testmohr")

@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings updates."""
    mode_map = {
        "Raw Mode": "r",
        "Informative Mode": "i",
        "Conversational Mode": "c"
    }
    
    cl.user_session.set("mode", mode_map[settings["mode"]])
    cl.user_session.set("tenant", settings["tenant"])
    
    await cl.Message(
        content=f"✅ Settings updated:\n- Mode: {settings['mode']}\n- Tenant: {settings['tenant']}"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle user messages."""
    logger.info(f"New message received: {message.content}")
    
    # Get settings from user session
    mode = cl.user_session.get("mode")
    tenant = cl.user_session.get("tenant")
    
    logger.debug(f"Session settings - Mode: {mode}, Tenant: {tenant}")
    
    if not mode or not isinstance(mode, str):
        mode = "i"
        logger.debug(f"Invalid mode, defaulting to: {mode}")
    
    if not tenant:
        logger.warning("No tenant selected")
        await cl.Message("⚠️ Please select a tenant first.").send()
        return
    
    loading_msg = cl.Message(content="⏳ Processing...")
    await loading_msg.send()
    
    try:
        # Initialize state
        initial_state = {
            'mode': mode,  # Now will be 'r' for "Raw Mode"
            'tenant_id': db.get_tenant_id(tenant),
            'question': message.content,
            'query': '',
            'valid': False,
            'result': '',
            'answer': '',
            'failed_queries': [],
            'column_names': []
        }
        
        logger.info("\n" + "="*50 + "\nINITIAL STATE:")
        for key, value in initial_state.items():
            logger.info(f"{key}: {value}")
        logger.info("="*50)
        
        if not initial_state['tenant_id']:
            logger.error(f"Invalid tenant ID for tenant: {tenant}")
            raise ValueError(f"Invalid tenant: {tenant}")
        
        # Initialize processors
        processors = [
            ('ProcessInput', ProcessInput(db.get_tenant_id)),
            ('QueryGeneration', QueryGeneration(db.schema_file_path, db.database, db.llm)),
            ('QueryValidation', QueryValidation(db.schema_file_path, db.llm)),
            ('QueryExecution', QueryExecution(db.database)),
            ('AnswerGeneration', AnswerGeneration(db))
        ]
        
        logger.info("Initialized all processors")
        
        # Process flow
        state = initial_state
        for processor_name, processor in processors:
            logger.info(f"\n{'='*50}\nPROCESSOR: {processor_name}")
            logger.info("INPUT STATE:")
            for key, value in state.items():
                logger.info(f"{key}: {value}")
            
            state = processor.process(state)
            
            logger.info("\nOUTPUT STATE:")
            for key, value in state.items():
                logger.info(f"{key}: {value}")
            logger.info("="*50)
            
            # Handle validation failure
            if not state['valid'] and isinstance(processor, QueryValidation):
                logger.warning("Query validation failed, attempting retry")
                await cl.Message("⚠️ Invalid query. Trying alternative approach...").send()
                
                # Retry query generation with failed queries
                logger.info("\nRETRYING QUERY GENERATION")
                state = processors[1][1].process(state)  # QueryGeneration
                logger.info(f"Retry query: {state['query']}")
                
                state = processor.process(state)      # QueryValidation again
                logger.info(f"Retry validation result - Valid: {state['valid']}")
                
                if not state['valid']:
                    logger.error("Query validation failed after retry")
                    await cl.Message("⚠️ Could not generate a valid query. Please rephrase your question.").send()
                    return
        
        logger.info("\nFINAL STATE:")
        for key, value in state.items():
            logger.info(f"{key}: {value}")
        logger.info("="*50)
        
        # Handle response based on mode
        if state['mode'] == 'r':
            logger.debug("Processing raw mode response")
            logger.debug(f"State before CSV creation - Result: {state['result'][:200]}...")
            logger.debug(f"State before CSV creation - Column names: {state['column_names']}")
            
            df, summary = db.create_csv(state['mode'], state['result'], state['column_names'])
            
            if df is not None:
                logger.info(f"Created DataFrame with shape: {df.shape}")
                logger.debug(f"DataFrame columns: {df.columns.tolist()}")
                logger.debug(f"First row of data: {df.iloc[0].to_dict()}")
                await send_data_response(df, summary, state['query'])
            else:
                logger.warning("Could not create DataFrame, sending raw answer")
                logger.debug(f"Raw result being sent: {state['result'][:200]}...")
                await cl.Message(content=str(state['result'])).send()
        else:
            logger.debug("Sending formatted response")
            await cl.Message(content=state['answer']).send()
            await cl.Message(
                content=f"🔍 Query used:\n```sql\n{state['query']}\n```"
            ).send()
            
        # Print token report to terminal after processing
        print_token_report()
            
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        await cl.Message(f"❌ Error: {str(e)}").send()
    finally:
        await loading_msg.remove()

@cl.on_stop
async def stop():
    await cl.Message("👋 Thanks for using MOHR AI Assistant! Have a great day!").send()

@cl.on_chat_end
async def end():
    await cl.Message("👋 Thanks for using MOHR AI Assistant! Have a great day!").send()

if __name__ == "__main__":
    cl.run()
