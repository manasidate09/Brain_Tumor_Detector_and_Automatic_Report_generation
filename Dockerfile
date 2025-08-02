# Base image with Python
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /Brain_Tumor_Detector_and_Automatic_Report_generation

# Install system dependencies (optional, tweak as needed)
RUN apt-get update && apt-get install -y git

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy entire project
COPY . .

# (Optional) Hugging Face token setup if using private models
# ENV HF_TOKEN=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

# Run your app (change to streamlit, flask, or your command)
CMD ["python", "brain_tumor_app.py"]
