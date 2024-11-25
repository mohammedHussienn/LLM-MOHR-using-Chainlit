import chainlit as cl
from chainlit.input_widget import Select
from database_backend import DatabaseBackend
import logging
import pandas as pd
from io import StringIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database backend
db = DatabaseBackend()

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
        initial_value="mendel-ai"  # Set default tenant
    )
    
    mode_select = Select(
        id="mode",
        label="Chat Mode",
        values=["Raw Mode", "Informative Mode", "Conversational Mode"],
        initial_value="Raw Mode"  # Set default mode
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
    try:
        # Get settings from session
        mode = cl.user_session.get("mode", "i")  # default to informative mode
        tenant = cl.user_session.get("tenant")
        
        if not tenant:
            await cl.Message("⚠️ Please select a tenant first.\n\nYou can do this by clicking on the settings gear in the left corner of the chat bar.").send()
            return
        
        # Show thinking message
        await cl.Message(content="🤔 Let me think about that...").send()
        
        # Get response from backend
        response = db.invoke_prompt(mode, message.content, tenant)

        # Handle response based on mode
        if mode == 'r':
            # Raw mode - display as table
            df, summary = db.create_csv(mode, response)
            
            if df is not None:
                # Create CSV string
                csv_string = df.to_csv(index=False)
                
                # Send the summary message
                await cl.Message(content=summary).send()
                
                # Add downloadable CSV file
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
            else:
                await cl.Message(content=response).send()
        else:
            # Informative or Conversational mode - display as text
            await cl.Message(content=response).send()
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        await cl.Message(f"❌ An error occurred: {str(e)}").send()

@cl.on_stop
async def stop():
    await cl.Message("👋 Thanks for using MOHR AI Assistant! Have a great day!").send()

if __name__ == "__main__":
    cl.run()
