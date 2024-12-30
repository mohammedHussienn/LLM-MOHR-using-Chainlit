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

### 1. Viewing Alert Data and analyzing it
#### Select the tenant you want to view
```
Tenant: mendel-ai (as an example)
```
#### Ask a question about the data
```
Question: show me all the employees' names and ids, alert types (with their meaning), and the total minutes of each alert (each alert and its minutes should be in separate columns (i want each employee to have one single row, with all the alerts having separate columns) for july 2024
```
#### The system will generate a query, execute it, and return the results in an excel file. 
#### You can then analyze the data in the excel file. Using three different methods:
##### 1. Text data analysis
```
Question: show me the names of the most late employees 
```
##### 2. Image visualization chart
```
Question: show me 5 most late people and order it from most late to least late and write their late hours at the end of each bar
```
##### 3. PDF visualization charts
```
Question: show me all the employees, there should be a legend showing the alert types in the page, also if there are zeros in some alert better to ignore that alert than leaving space for it. i want every alert to be separate in its own bar and make sure the pages are consequent and the graphs are readable
```