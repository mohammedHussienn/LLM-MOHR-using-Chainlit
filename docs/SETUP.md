# Setup Guide

## Prerequisites
- Python 3.8 or higher
- SQL Server with appropriate access
- Azure Data Studio (for database management and monitoring)
- OpenAI API key

## Project Structure
```
mohr-ai/
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── database_backend.py
│   └── config/
│       ├── newSchema.txt
│       └── examples.txt
├── .chainlit/
│   └── config.toml
├── docs/
│   ├── SETUP.md
│   └── API.md
├── .files/          # Generated during runtime
├── .env             # You need to create this
├── requirements.txt
├── README.md
├── chainlit.md
└── Dockerfile
```

## Environment Setup
1. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file with the following variables:
```env
OPENAI_API_KEY=your_api_key
SQL_SERVER=your_server
SQL_DATABASE=your_database
SQL_USERNAME=your_username
SQL_PASSWORD=your_password
```

## Database Setup
1. Install and configure Azure Data Studio
2. Connect to your SQL Server instance
3. Ensure proper permissions are set for the database user

## Running the Application
From the project root directory:
```bash
chainlit run src/app.py
``` 