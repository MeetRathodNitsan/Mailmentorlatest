# Base image with Python
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy everything from your project into the Docker image
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
