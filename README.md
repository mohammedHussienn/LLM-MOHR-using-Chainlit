# MOHR AI Assistant

An intelligent chatbot that provides SQL query assistance, data analysis, and visualization capabilities for the MOHR database system.

## 🚀 Functionality

### Core Features
- **SQL Query Generation**: Automatically generates SQL queries based on natural language questions
- **Data Analysis**: Performs analysis on query results using pandas DataFrame operations
- **Visualization**: Creates custom visualizations in both image and PDF formats
- **Multi-tenant Support**: Handles different tenant contexts securely
- **Interactive Mode Switching**: Seamlessly switch between:
  - Raw data querying
  - Data analysis
  - Visualization generation

### Key Capabilities
- Natural language to SQL translation
- Secure query validation
- Error handling and query retries
- Dynamic visualization generation
- Multi-format data export (Excel, PDF, PNG)
- Context-aware tenant management

## 🏗️ Architecture

### Component Structure
1. **Database Backend (`database_backend.py`)**
   - `DatabaseBackend`: Core database operations and LLM interactions
   - `QueryGeneration`: SQL query generation from natural language
   - `QueryValidation`: Security and syntax validation
   - `QueryExecution`: Safe query execution
   - `GraphGenerationAgent`: Visualization code generation
   - `TempFileManager`: File management for visualizations

2. **Application Layer (`app.py`)**
   - Chainlit web interface
   - Session management
   - User interaction handling
   - Mode switching logic
   - File handling and response formatting

### Data Flow
1. User inputs question via Chainlit interface
2. System determines operation mode (raw/analysis/visualization)
3. Query generation and validation (if in raw mode)
4. Data processing and DataFrame creation
5. Analysis or visualization generation
6. Result presentation to user

### Security Features
- Tenant isolation using `SET CONTEXT_INFO`
- Query validation to prevent harmful operations
- Secure file management in `.files` directory
- Environment variable based configuration

## 📝 Example Cases

### 1. Basic Data Query
```

## 📚 Documentation

For more detailed information about specific components:
- See `database_backend.py` for backend logic
- See `app.py` for application flow
- Check `examples.txt` for query examples
- Refer to `newSchema.txt` for database schema
