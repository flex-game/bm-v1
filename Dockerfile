# Explicitly set platform
FROM --platform=linux/amd64 tensorflow/tensorflow:2.14.0

# Add SageMaker capability label
LABEL com.amazonaws.sagemaker.capabilities=["train","serve"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install all packages in one layer to reduce image size
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

# Pre-download weights without executing TensorFlow code
RUN mkdir -p /root/.keras/models/ && \
    wget https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /root/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

# Set working directory
WORKDIR /opt/ml/code

# Copy all necessary files
COPY . .

# Make scripts executable
RUN chmod +x train.py model_train.py

# Set entrypoint for SageMaker
ENTRYPOINT ["python3", "model_train.py"] 