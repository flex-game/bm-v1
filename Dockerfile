FROM tensorflow/tensorflow:2.14.0

# Add SageMaker capability label
LABEL com.amazonaws.sagemaker.capabilities=["train","serve"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (split into separate RUN commands for better debugging)
RUN pip3 install --no-cache-dir \
    sagemaker-training \
    boto3 \
    numpy \
    pandas \
    pillow \
    scikit-learn \
    python-dotenv \
    requests

# Install TensorFlow-related packages separately
RUN pip3 install --no-cache-dir \
    tensorflow-hub \
    tensorflow-text

# Download ResNet50 weights in a separate step
RUN python3 -c "import tensorflow as tf; tf.keras.applications.ResNet50(weights='imagenet', include_top=False)"

# Set working directory
WORKDIR /opt/ml/code

# Copy all necessary files
COPY . .

# Make scripts executable
RUN chmod +x train.py model_train.py

# Set entrypoint for SageMaker
ENTRYPOINT ["python3", "model_train.py"] 