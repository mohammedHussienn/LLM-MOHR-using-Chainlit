# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    unixodbc \
    unixodbc-dev \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install SQL Server ODBC Driver
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p src/config .chainlit .files docs

# Copy application files
COPY src/ ./src/
COPY .chainlit/ ./.chainlit/
COPY docs/ ./docs/
COPY README.md chainlit.md ./

# Ensure proper permissions
RUN chmod -R 755 .files

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose the port Chainlit runs on
EXPOSE 8000

# Command to run the application
CMD ["chainlit", "run", "src/app.py", "--host", "0.0.0.0", "--port", "8000"] 