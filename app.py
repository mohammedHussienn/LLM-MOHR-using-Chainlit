import chainlit as cl
from chainlit.input_widget import Select
from database_backend import DatabaseBackend, State
import logging
import pandas as pd
from io import StringIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize database backend
db = DatabaseBackend(schema_file_path="databaseSchema.txt")

@cl.on_chat_start
async def start():
    # Send welcome message with markdown formatting
    await cl.Message(
        content="""👋 Welcome to MOHR AI Assistant!

I'm here to help you with your queries. Here's what you can do:
- Ask questions about data
- Get information in different formats
- Have natural conversations about the data

Please select your preferences below to get started!
""").send()
    
    # Get and sort tenants
    tenants = sorted(db.get_all_tenants())
    
    # Create selectors with new defaults
    tenant_select = Select(
        id="tenant",
        label="Select Tenant",
        values=tenants,
        initial_value="testmohr"  # Set default tenant
    )
    
    mode_select = Select(
        id="mode",
        label="Chat Mode",
        values=["Raw Mode", "Informative Mode", "Conversational Mode"],
        initial_value="Informative Mode"  # Set default mode
    )
    
    # Send settings
    settings = await cl.ChatSettings([mode_select, tenant_select]).send()
    
    # Set default settings in session
    cl.user_session.set("mode", "r")  # default to raw mode
    cl.user_session.set("tenant", "mendel-ai")  # default tenant

@cl.on_settings_update
async def setup_agent(settings):
    # Map friendly names to mode codes
    mode_map = {
        "Raw Mode": "r",
        "Informative Mode": "i",
        "Conversational Mode": "c"
    }
    
    # Store settings in session
    cl.user_session.set("mode", mode_map[settings["mode"]])
    cl.user_session.set("tenant", settings["tenant"])
    
    # Send confirmation message
    await cl.Message(
        content=f"✅ Settings updated:\n- Mode: {settings['mode']}\n- Tenant: {settings['tenant']}"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    logger.info("=== New Message Received ===")
    logger.info(f"Message content: {message.content}")
    
    try:
        # Get settings
        mode = cl.user_session.get("mode", "i")
        tenant = cl.user_session.get("tenant")
        logger.info(f"Current settings - Mode: {mode}, Tenant: {tenant}")
        
        if not tenant:
            logger.warning("No tenant selected")
            await cl.Message("⚠️ Please select a tenant first.").send()
            return
        
        logger.info("Initializing processing messages...")
        thinking_msg = cl.Message(content="🤔 Let me think about that...")
        await thinking_msg.send()
        
        processing_msg = cl.Message(content="⚙️ Processing your request...")
        await processing_msg.send()
        
        try:
            # Initialize state
            logger.info("Getting tenant ID and initializing state...")
            tenant_id = db.get_tenant_id(tenant)
            logger.info(f"Tenant ID retrieved: {tenant_id}")
            
            state: State = {
                'tenant_id': tenant_id,
                'question': message.content,
                'query': '',
                'result': '',
                'answer': ''
            }
            logger.info("State initialized")

            # Get and validate query
            logger.info("Generating query...")
            query_result = db.get_query(state)
            logger.info("Validating query...")
            if not db.validate_query(query_result['query']):
                logger.warning("Query validation failed")
                await thinking_msg.remove()
                await processing_msg.remove()
                await cl.Message(content="⚠️ Invalid query generated. Please rephrase.").send()
                return

            # Update state and get response
            logger.info("Getting response from database...")
            response = db.invoke_prompt(mode, message.content, tenant)
            logger.info("Response received")

        except Exception as query_error:
            logger.error(f"Query error: {query_error}")
            await thinking_msg.remove()
            await processing_msg.remove()
            
            if "iteration limit" in str(query_error).lower():
                logger.warning("Iteration limit exceeded")
                await cl.Message(
                    content="⚠️ Query too complex. Please simplify."
                ).send()
                return
            else:
                raise query_error
            
        logger.info("Removing processing messages...")
        await thinking_msg.remove()
        await processing_msg.remove()
        
        # Handle response based on mode
        logger.info(f"Processing response for mode: {mode}")
        if mode == 'r':
            logger.info("Processing raw mode response...")
            df, summary = db.create_csv(mode, response)
            
            if df is not None:
                logger.info(f"DataFrame created successfully. Shape: {df.shape}")
                csv_string = df.to_csv(index=False)
                
                await cl.Message(content=summary).send()
                logger.info("Summary sent")
                
                await cl.Message(
                    content="You can download the complete dataset here:",
                    elements=[
                        cl.File(
                            name="data.csv",
                            content=csv_string.encode(),
                            mime="text/csv"
                        )
                    ]
                ).send()
                logger.info("CSV file sent")

                if state['query']:
                    await cl.Message(content=f"Query used:\n```sql\n{state['query']}\n```").send()
                    logger.info("Query display sent")
            else:
                logger.warning("No DataFrame created, sending raw response")
                await cl.Message(content=response).send()
        else:
            logger.info("Sending text response")
            await cl.Message(content=response).send()
            
            if state['query']:
                await cl.Message(content=f"Query used:\n```sql\n{state['query']}\n```").send()
                logger.info("Query display sent")
            
    except Exception as e:
        logger.error(f"Error in main handler: {str(e)}", exc_info=True)
        await cl.Message(f"❌ An error occurred: {str(e)}").send()

    logger.info("=== Message Processing Complete ===\n")

@cl.on_stop
async def stop():
    await cl.Message("👋 Thanks for using MOHR AI Assistant! Have a great day!").send()

if __name__ == "__main__":
    cl.run()
