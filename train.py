import tensorflow as tf
import logging
import boto3
import json
import numpy as np
from io import BytesIO
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from utils.text_preprocessing import preprocess_texts
from utils.s3_utils import s3_list_bucket_objects, s3_get_matching_files, s3_verify_bucket_access
from utils.actions import prepare_action_labels

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    s3_verify_bucket_access()
    # Configure and start SageMaker training job

    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Get number of output classes
    _, num_actions = prepare_action_labels()
    
    # Define hyperparameters
    hyperparameters = {
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_actions': num_actions,
        'embedding_dim': 50,
        'max_sequence_length': 50,
        'num_words': 10000
    }
    
    # Configure estimator
    estimator = TensorFlow(
        entry_point='model_train.py',  # Your training script
        role=role,
        instance_count=1,
        instance_type='ml.p3.2xlarge',  # GPU instance
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