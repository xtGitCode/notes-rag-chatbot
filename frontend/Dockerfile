# Use the slim version of Python 3.10
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . .

# Install Streamlit dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "steamlit_app.py", "--server.port=8501", "--server.enableCORS=false", "--server.address=0.0.0.0"]
