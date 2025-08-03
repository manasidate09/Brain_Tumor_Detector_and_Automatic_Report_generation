# Use lightweight Python image
FROM python3.10-slim

# Install required system dependencies for OpenCV and others
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY brain_tumor_app.py .
COPY brain_tumor_classifier.h5 .
COPY tumor_segmentation_model.h5 .

RUN ls -lh brain_tumor_classifier.h5
RUN ls -lh tumor_segmentation_model.h5

# Copy the rest of the code
COPY . .

EXPOSE 8501

# Run the app
CMD [streamlit, run, brain_tumor_app.py, --server.port=8501, --server.address=0.0.0.0]






