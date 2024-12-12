import logging
from dotenv import load_dotenv
from utils.s3_utils import s3_verify_bucket_access
from utils.actions import prepare_action_labels
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sagemaker
import os
import argparse
from sagemaker.tensorflow import TensorFlow

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

# Check both CLI argument and environment variable
force_refresh = args.refresh_actions or os.environ.get('REFRESH_ACTIONS', '').lower() == 'true'

if __name__ == "__main__":
    s3_verify_bucket_access()
    # Configure and start SageMaker training job

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
        entry_point='model_train.py',  # Your training script
        role=role,
        instance_count=1,
        instance_type='ml.c4.xlarge',
        framework_version='2.11',
        py_version='py39',
        hyperparameters=hyperparameters,
        output_path='s3://bm-v1-model/trained_models'
    )
    
    # Define data channels
    data_channels = {
        'training': f's3://bm-v1-training-images',
        'text': f's3://bm-v1-training-text',
        'actions': f's3://bm-v1-training-actions'
    }
    
    # Start training
    estimator.fit(data_channels)