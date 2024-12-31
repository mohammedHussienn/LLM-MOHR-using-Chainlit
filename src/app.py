import chainlit as cl
from chainlit.input_widget import Select
from database_backend import DatabaseBackend, State, ProcessInput, QueryGeneration, QueryValidation, QueryExecution, AnswerGeneration, DataFrameAgent, GraphGenerationAgent, TempFileManager
import logging
import pandas as pd
from typing import Optional, cast, Literal
import json
from datetime import datetime
import io
import contextlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import traceback
import os
from pathlib import Path

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

# Get the absolute path to the config directory
CONFIG_DIR = Path(__file__).parent / "config"

# Initialize database backend with correct path
db = DatabaseBackend(
    schema_file_path=str(CONFIG_DIR / "newSchema.txt")
)

# Add these global variables
temp_manager = TempFileManager()

async def send_data_response(df: pd.DataFrame, summary: str, query: Optional[str] = None):
    """Send formatted data response to chat."""
    if len(df) > 0:
        try:
            # Send the summary and query first
            message = f"{summary}\n\n"
            if query:
                message += f"Query used:\n```sql\n{query}\n```"
            
            await cl.Message(content=message).send()

            # Create Excel file in memory
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            
            file_name = f"data_export.xlsx"
            await cl.Message(
                content=f"📥 Download complete dataset ({len(df)} records):",
                elements=[
                    cl.File(
                        name=file_name,
                        content=excel_buffer.getvalue(),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                ],
                actions=await get_mode_buttons('r')
            ).send()
            
            # Store the current dataframe in the user session
            cl.user_session.set("current_df", df)
            
        except Exception as e:
            logger.error(f"Error sending data response: {str(e)}", exc_info=True)
            await cl.Message(content=f"Error displaying data: {str(e)}").send()
    else:
        await cl.Message(content="No records found.").send()

async def send_analysis_response(answer: str):
    """Send analysis response with mode-switch button."""
    await cl.Message(
        content=answer,
        actions=[
            cl.Action(
                name="get_new_data",
                value="new_data",
                label="Get New Data",
                description="Switch to query mode"
            )
        ]
    ).send()

@cl.action_callback("get_new_data")
async def on_get_new_data(action):
    """Handle get new data button click"""
    cl.user_session.set("mode", "r")
    await cl.Message("Switched to query mode. Please ask your new data query.").send()

@cl.action_callback("ask_current_dataset")
async def on_ask_current_dataset(action):
    """Handle ask current dataset button click"""
    df = cl.user_session.get("current_df")
    if df is not None:
        cl.user_session.set("mode", "a")
        await cl.Message(
            content=f"📊 Analysis Mode: You can ask questions about the current dataset ({len(df)} records)\n"
            f"Available columns: {', '.join(df.columns.tolist())}\n\n"
            "What would you like to do with this data?",
            actions=[
                cl.Action(
                    name="mode_analysis",
                    value="analysis",
                    label="Analyze Data",
                    description="Ask questions about the data"
                ),
                cl.Action(
                    name="mode_image",
                    value="image",
                    label="Generate Graph Image",
                    description="Create visualization as image"
                ),
                cl.Action(
                    name="mode_pdf",
                    value="pdf",
                    label="Generate PDF Graph",
                    description="Create visualization as PDF"
                )
            ]
        ).send()
    else:
        await cl.Message("No dataset is currently loaded. Please query for data first.").send()

@cl.action_callback("mode_analysis")
async def on_mode_analysis(action):
    cl.user_session.set("mode", "a")
    cl.user_session.set("analysis_mode", "text")
    await cl.Message("📊 Analysis Mode: Ask your question about the data.").send()

@cl.action_callback("mode_image")
async def on_mode_image(action):
    cl.user_session.set("mode", "a")
    cl.user_session.set("analysis_mode", "image")
    await cl.Message("📈 Image Visualization Mode: Describe the graph you want to create.").send()

@cl.action_callback("mode_pdf")
async def on_mode_pdf(action):
    cl.user_session.set("mode", "a")
    cl.user_session.set("analysis_mode", "pdf")
    await cl.Message("📑 PDF Visualization Mode: Describe the graph you want to create.").send()

@cl.on_chat_start
async def start():
    """Initialize chat session."""
    await cl.Message(
        content="""👋 Welcome to MOHR AI Assistant!

I'm here to help you with your queries. I'll provide the data in a downloadable Excel format.

Please select your tenant to get started!
""").send()
    
    # Setup chat settings - only tenant selection remains
    tenants = sorted(db.get_all_tenants())
    settings = await cl.ChatSettings([
        Select(
            id="tenant",
            label="Select Tenant",
            values=tenants,
            initial_value="testmohr"
        )
    ]).send()
    
    # Set default session values - only mode 'r' and tenant
    cl.user_session.set("mode", "r")  # Always raw mode
    cl.user_session.set("tenant", "testmohr")

@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings updates."""
    cl.user_session.set("tenant", settings["tenant"])
    
    await cl.Message(
        content=f"✅ Settings updated:\n- Tenant: {settings['tenant']}"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle user messages."""
    mode = cl.user_session.get("mode", "r") or "r"
    analysis_mode = cl.user_session.get("analysis_mode", "text")
    loading_msg = cl.Message(content="⏳ Processing...")
    await loading_msg.send()
    
    try:
        if mode == "a":
            df = cl.user_session.get("current_df")
            if df is None:
                await cl.Message("No dataset available to analyze. Please query for data first.").send()
                return
            
            state = cast(State, {
                'mode': mode,
                'analysis_mode': analysis_mode,
                'tenant_id': cl.user_session.get("tenant_id"),
                'question': message.content,
                'query': '',
                'column_names': [],
                'valid': True,
                'result': '',
                'answer': '',
                'failed_queries': [],
                'current_df': df
            })
            
            if analysis_mode in ('image', 'pdf'):
                viz_agent = GraphGenerationAgent(df)
                result_state = viz_agent.process(state, cast(Literal['image', 'pdf'], analysis_mode))
                
                try:
                    # Execute the generated code
                    if analysis_mode == 'pdf':
                        pdf_path = temp_manager.get_file_path('.pdf')
                        with PdfPages(pdf_path) as pdf:
                            exec(result_state['answer'])
                            # Save all figures that were generated
                            for fig in plt.get_fignums():
                                pdf.savefig(fig)
                                plt.close(fig)
                        
                        await cl.Message(
                            content=f"""📊 Data Analysis Summary:
{result_state.get('summary', 'No summary available.')}

📑 Generated PDF visualization:""",
                            elements=[
                                cl.File(name="visualization.pdf", path=pdf_path)
                            ],
                            actions=await get_mode_buttons(mode)
                        ).send()
                        
                    else:  # image mode
                        img_path = temp_manager.get_file_path('.png')
                        exec(result_state['answer'])
                        plt.savefig(img_path, bbox_inches='tight', dpi=300)
                        plt.close()
                        
                        await cl.Message(
                            content=f"""📊 Data Analysis Summary:
{result_state.get('summary', 'No summary available.')}

📈 Generated visualization:""",
                            elements=[
                                cl.Image(name="visualization.png", path=img_path)
                            ],
                            actions=await get_mode_buttons(mode)
                        ).send()
                    
                except Exception as e:
                    logger.error(f"Error executing visualization code: {e}")
                    await cl.Message(
                        content=f"❌ Error creating visualization: {str(e)}\n\nGenerated code:\n```python\n{result_state['answer']}\n```"
                    ).send()
            else:
                # Text analysis mode
                df_agent = DataFrameAgent()
                result_state = df_agent.process(state)
                await cl.Message(
                    content=result_state['answer'],
                    actions=await get_mode_buttons(mode)
                ).send()
            
            return  # Add this return statement to prevent falling through to raw mode
            
        # Raw mode handling
        tenant = cl.user_session.get("tenant")
        if not tenant:
            await cl.Message("⚠️ Please select a tenant first.").send()
            return

        # Initialize state
        initial_state = cast(State, {
            'mode': 'r',
            'tenant_id': db.get_tenant_id(tenant),
            'question': message.content,
            'query': '',
            'valid': False,
            'result': '',
            'answer': '',
            'failed_queries': [],
            'column_names': [],
            'current_df': None
        })

        # Initialize processors
        processors = [
            ('ProcessInput', ProcessInput(db.get_tenant_id)),
            ('QueryGeneration', QueryGeneration(db.schema_file_path, db.database, db.llm)),
            ('QueryValidation', QueryValidation(db.schema_file_path, db.llm)),
            ('QueryExecution', QueryExecution(db.database))
        ]

        # Process flow with retries
        state = initial_state
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            should_retry = False
            
            for processor_name, processor in processors:
                state = processor.process(state)
                
                if not state['valid']:
                    if retry_count >= max_retries - 1:
                        await cl.Message("⚠️ Could not generate a valid query after multiple attempts. Please rephrase your question.").send()
                        return
                    
                    if isinstance(processor, (QueryValidation, QueryExecution)):
                        await cl.Message("⚠️ No results found. Trying alternative approach...").send()
                        should_retry = True
                        break
            
            if should_retry:
                retry_count += 1
                continue
                
            if state['valid'] and state['result']:
                break
                
            retry_count += 1

        # Create DataFrame and send response
        df, summary = db.create_csv('r', state['result'], state['column_names'])
        if df is not None:
            await send_data_response(df, summary, state['query'])
        else:
            await cl.Message(content="No records found.").send()
            
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
    temp_manager.cleanup()
    await cl.Message("👋 Thanks for using MOHR AI Assistant! Have a great day!").send()

async def get_mode_buttons(current_mode: str):
    """Get appropriate action buttons based on current mode."""
    # Always include Get New Data unless we're in raw mode
    buttons = []
    if current_mode != 'r':
        buttons.append(
            cl.Action(
                name="get_new_data",
                value="new_data",
                label="Get New Data",
                description="Query for new data"
            )
        )
    
    # Add Analysis button if we're not in analysis text mode
    if current_mode != "a" or cl.user_session.get("analysis_mode") != "text":
        buttons.append(
            cl.Action(
                name="mode_analysis",
                value="analysis",
                label="Analyze Data",
                description="Ask questions about the data"
            )
        )
    
    # Add Image button if we're not in image mode
    if current_mode != "a" or cl.user_session.get("analysis_mode") != "image":
        buttons.append(
            cl.Action(
                name="mode_image",
                value="image",
                label="Generate Graph Image",
                description="Create visualization as image"
            )
        )
    
    # Add PDF button if we're not in PDF mode
    if current_mode != "a" or cl.user_session.get("analysis_mode") != "pdf":
        buttons.append(
            cl.Action(
                name="mode_pdf",
                value="pdf",
                label="Generate PDF Graph",
                description="Create visualization as PDF"
            )
        )
    
    return buttons

if __name__ == "__main__":
    cl.run()
