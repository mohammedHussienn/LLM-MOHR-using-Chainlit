import chainlit as cl
from chainlit.input_widget import Select
from database_backend import DatabaseBackend, State
import logging
import pandas as pd
from typing import Optional

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
db = DatabaseBackend(schema_file_path="databaseSchema.txt")

async def format_table_preview(df: pd.DataFrame) -> str:
    """Format DataFrame for display in chat."""
    preview_df = df.head(10)
    if TABULATE_INSTALLED:
        return preview_df.to_markdown(index=False)
    return "\n".join([
        "| " + " | ".join(str(x) for x in row) + " |"
        for row in [preview_df.columns.tolist()] + preview_df.values.tolist()
    ])

async def send_data_response(df: pd.DataFrame, summary: str, query: Optional[str] = None):
    """Send formatted data response to chat."""
    if len(df) > 0:
        table_str = await format_table_preview(df)
        await cl.Message(
            content=f"{summary}\n\nPreview of first 10 records:\n```\n{table_str}\n```"
        ).send()

    # Send downloadable CSV
    file_data = df.to_csv(index=False).encode()
    await cl.Message(
        content="📥 Download complete dataset:",
        elements=[
            cl.File(
                name="data.csv",
                content=file_data,
                mime="text/csv"
            )
        ]
    ).send()

    # Show query if available
    if query:
        await cl.Message(
            content=f"🔍 Query used:\n```sql\n{query}\n```"
        ).send()

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
    
    # Get current settings
    mode = cl.user_session.get("mode", "i")  # Default to informative mode
    tenant = cl.user_session.get("tenant")
    
    if not mode or not isinstance(mode, str):
        mode = "i"  # Default to informative mode if invalid
    
    if not tenant:
        await cl.Message("⚠️ Please select a tenant first.").send()
        return
    
    # Show processing status
    loading_msg = cl.Message(content="⏳ Processing...")
    await loading_msg.send()
    
    try:
        # Initialize state
        tenant_id = db.get_tenant_id(tenant)
        if not tenant_id:
            raise ValueError(f"Invalid tenant: {tenant}")
        
        state: State = {
            'tenant_id': tenant_id,
            'question': message.content,
            'valid': False,
            'query': '',
            'result': '',
            'answer': '',
            'failed_queries': []
        }
        
        # Generate and validate query
        query = db.query_chain.generate_query(
            question=state['question'],
            tenant_id=tenant_id,
            failed_queries=[]
        )
        
        is_valid, reason = db.validator.validate(query)
        if not is_valid:
            await cl.Message(f"⚠️ Invalid query: {reason}").send()
            return
        
        # Get response
        response = db.invoke_prompt(mode, message.content, tenant)
        
        # Process response based on mode
        if mode == 'r':
            df, summary = db.create_csv(mode, response)
            if df is not None:
                await send_data_response(df, summary, query)
            else:
                await cl.Message(content=str(response)).send()
        else:
            await cl.Message(content=response).send()
            await cl.Message(
                content=f"🔍 Query used:\n```sql\n{query}\n```"
            ).send()
            
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        await cl.Message(f"❌ Error: {str(e)}").send()
    finally:
        await loading_msg.remove()

@cl.on_stop
async def stop():
    await cl.Message("👋 Thanks for using MOHR AI Assistant! Have a great day!").send()

if __name__ == "__main__":
    cl.run()
