# API Documentation

## Database Backend

### Class: DatabaseBackend
Main class handling database operations and LLM interactions.

#### Methods:
- `generate_query(question: str) -> str`: Generates SQL query from natural language
- `execute_query(query: str) -> pd.DataFrame`: Executes SQL query safely
- `generate_visualization(data: pd.DataFrame, request: str)`: Creates visualizations
- `set_tenant_context(tenant_id: str)`: Sets the tenant context for queries

#### Database Connection
The system uses ODBC drivers to connect to SQL Server. Make sure your Azure Data Studio connection string matches the format in your `.env` file.

## Query Generation
Queries are generated using OpenAI's language models with specific prompts that ensure:
- SQL Server compatibility
- Proper tenant isolation
- Safe query construction

## Visualization Generation
Supports multiple visualization types:
- Bar charts
- Line graphs
- Pie charts
- Custom PDF reports

### File Management
Visualizations are stored in the `.files` directory with proper cleanup mechanisms. 