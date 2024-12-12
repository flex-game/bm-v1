FROM tensorflow/tensorflow:2.14.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    sagemaker-training \
    boto3 \
    numpy \
    pandas \
    pillow \
    scikit-learn \
    python-dotenv \
    requests \
    tensorflow-hub \
    tensorflow-text

# Set working directory
WORKDIR /opt/ml/code

# Copy training code
COPY . .

# Make train script executable
RUN chmod +x train.py

# Set entrypoint
ENTRYPOINT ["python3"] 