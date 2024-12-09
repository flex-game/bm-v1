import tensorflow as tf
import logging
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
import sagemaker
from sagemaker.tensorflow import TensorFlow
import json
import numpy as np
from io import BytesIO
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from utils.text_preprocessing import preprocess_texts

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def list_bucket_objects(bucket):
        objects = []
        paginator = s3_client.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=bucket):
                if 'Contents' in page:
                    # Extract just the filenames without paths
                    filenames = [Path(obj['Key']).stem for obj in page['Contents']]
                    objects.extend(filenames)
            return set(objects)
        except ClientError as e:
            logger.error(f"Error accessing bucket {bucket}: {str(e)}")
            raise

def get_matching_files_from_s3(image_bucket, text_bucket, actions_bucket):
    """Get lists of matching files from all three buckets"""
    s3_client = boto3.client('s3')

    # Get all filenames (without extensions) from each bucket
    image_files = list_bucket_objects(image_bucket)
    text_files = list_bucket_objects(text_bucket)
    action_files = list_bucket_objects(actions_bucket)
    
    # Find common files across all buckets
    common_files = sorted(list(image_files & text_files & action_files))
    
    if not common_files:
        raise ValueError("No matching files found across all three buckets")
    
    logger.info(f"Found {len(common_files)} matching files across all buckets")
    return common_files

def prepare_action_labels():
    # Extract and prepare unique action labels from training files

    s3_client = boto3.client('s3')
    actions_bucket = 'bm-v1-training-actions'
    unique_actions = set()
    
    # List all files in actions bucket
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=actions_bucket):
        for obj in page['Contents']:
            response = s3_client.get_object(Bucket=actions_bucket, Key=obj['Key'])
            action_data = json.loads(response['Body'].read().decode('utf-8'))
            
            # Add each action to our set of unique actions
            for action in action_data['actions_by_player']:
                unique_actions.add(action)
    
    # Convert to sorted list for consistent ordering
    action_list = sorted(list(unique_actions))
    action_mapping = {action: idx for idx, action in enumerate(action_list)}
    
    # Save action mapping to S3
    s3_client.put_object(
        Bucket='bm-v1-model',
        Key='action_mapping.json',
        Body=json.dumps(action_mapping)
    )
    
    return action_mapping, len(action_list)

def verify_bucket_access():
    """Verify access to all required S3 buckets"""
    s3_client = boto3.client('s3')
    buckets_to_check = {
        'read': ['bm-v1-training-images', 'bm-v1-training-text', 'bm-v1-training-actions'],
        'write': ['bm-v1-model']
    }
    
    for bucket in buckets_to_check['read']:
        try:
            s3_client.head_bucket(Bucket=bucket)
            # Test list operation
            s3_client.list_objects_v2(Bucket=bucket, MaxKeys=1)
            logger.info(f"Successfully verified read access to {bucket}")
        except Exception as e:
            raise Exception(f"Failed to access bucket {bucket}: {str(e)}")
    
    for bucket in buckets_to_check['write']:
        try:
            s3_client.head_bucket(Bucket=bucket)
            # Test write operation with a small test file
            s3_client.put_object(
                Bucket=bucket,
                Key='test_write_access.txt',
                Body='test'
            )
            s3_client.delete_object(
                Bucket=bucket,
                Key='test_write_access.txt'
            )
            logger.info(f"Successfully verified write access to {bucket}")
        except Exception as e:
            raise Exception(f"Failed to write to bucket {bucket}: {str(e)}")

if __name__ == "__main__":
    verify_bucket_access()
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