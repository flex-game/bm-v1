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

# Copy specific files needed for training
COPY model_train.py preprocessing.py utils/ /opt/ml/code/

# Make train script executable
COPY . .
RUN chmod +x train.py
RUN chmod +x model_train.py

# Set entrypoint
ENTRYPOINT ["python3"] 