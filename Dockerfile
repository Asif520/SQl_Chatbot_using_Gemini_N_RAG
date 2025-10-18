# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /gemini_sql_rag_1_ext

# Copy everything into the container
COPY . /gemini_sql_rag_1_ext

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Streamlit
EXPOSE 7860

# Run the app
CMD ["streamlit", "run", "gemini_sql_rag_1_ext.py", "--server.port=7860", "--server.address=0.0.0.0"]
