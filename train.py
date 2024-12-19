import logging
from dotenv import load_dotenv
from utils.s3_utils import s3_verify_bucket_access
from utils.actions import prepare_action_labels
import tensorflow as tf
import sagemaker
import os
import argparse
from sagemaker.tensorflow import TensorFlow
import psutil
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Add argument parsing
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--refresh-actions', action='store_true', 
                   help='Force refresh of action mappings')
args = parser.parse_args()

# Check both CLI argument and environment variables
force_refresh = args.refresh_actions or os.environ.get('REFRESH_ACTIONS', '').lower() == 'true'

def monitor_resources(start_time=None):
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = f"Elapsed: {elapsed:.1f}s"
    else:
        elapsed_str = "Initial check"

    logging.info(f"""
    Resource Usage ({elapsed_str}):
    Memory:
    - Total: {memory.total / (1024**3):.2f} GB
    - Available: {memory.available / (1024**3):.2f} GB
    - Used: {memory.used / (1024**3):.2f} GB
    - Percentage: {memory.percent}%
    CPU Usage: {cpu_percent}%
    """)
    return memory.used / (1024**3)  # Return GB used for tracking peak

if __name__ == "__main__":
    start_time = time.time()
    peak_memory = 0
    
    # Initial resource check
    initial_memory = monitor_resources()
    peak_memory = max(peak_memory, initial_memory)
    
    s3_verify_bucket_access()
    
    # Get the custom container image URI
    image_uri = os.getenv('SAGEMAKER_IMAGE_URI')
    if not image_uri:
        logger.error("SAGEMAKER_IMAGE_URI not found in environment variables")
        raise ValueError("SAGEMAKER_IMAGE_URI must be set")
    
    logger.info(f"Using Docker image: {image_uri}")
    
    sagemaker_session = sagemaker.Session()
    role = os.getenv('SAGEMAKER_ROLE_ARN')
    if not role:
        raise ValueError("SAGEMAKER_ROLE_ARN not found in environment variables")
    
    # Get number of output classes
    _, num_actions = prepare_action_labels(force_refresh=force_refresh)
    
    # Define hyperparameters
    hyperparameters = {
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_actions': num_actions,
        'embedding_dim': 50,
        'max_sequence_length': 50,
        'num_words': 2500
    }
    
    # Configure estimator
    estimator = TensorFlow(
        entry_point='model_train.py',
        source_dir='.',
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        framework_version='2.14',
        py_version='py310',
        image_uri=image_uri,
        hyperparameters=hyperparameters,
        output_path='s3://bm-v1-model/trained_models',
        code_location='s3://bm-v1-model/code',
        model_dir='s3://bm-v1-model/trained_models'
    )
    
    estimator.dependencies = ['code/']
    
    # Define data channels
    data_channels = {
        'training': f's3://bm-v1-training-images',
        'text': f's3://bm-v1-training-text',
        'actions': f's3://bm-v1-training-actions'
    }
    
    # Start training
    estimator.fit(data_channels)