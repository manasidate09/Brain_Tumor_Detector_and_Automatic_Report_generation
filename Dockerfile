# Use a light base Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /brain_tumor_app

# Copy requirements and install them
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (including 900MB models)
COPY . .

# Expose port if you're using Streamlit or Flask
EXPOSE 8501

# Default command to run the app
CMD ["python", "brain_tumor_app.py"]
