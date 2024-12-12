import boto3
import logging
from botocore.exceptions import ClientError
from pathlib import Path
import json
from utils.actions import parse_action_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def s3_list_bucket_objects(bucket):
    """List objects in an S3 bucket."""
    s3_client = boto3.client('s3')
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

def s3_get_matching_files(image_bucket, text_bucket, actions_bucket):
    """Get lists of matching files from all three buckets."""
    image_files = s3_list_bucket_objects(image_bucket)
    text_files = s3_list_bucket_objects(text_bucket)
    action_files = s3_list_bucket_objects(actions_bucket)
    
    common_files = sorted(list(image_files & text_files & action_files))
    
    if not common_files:
        raise ValueError("No matching files found across all three buckets")
    
    logger.info(f"Found {len(common_files)} matching files across all buckets")
    return common_files

def s3_verify_bucket_access():
    """Verify access to all required S3 buckets."""
    s3_client = boto3.client('s3')
    buckets_to_check = {
        'read': ['bm-v1-training-images', 'bm-v1-training-text', 'bm-v1-training-actions'],
        'write': ['bm-v1-model']
    }
    
    for bucket in buckets_to_check['read']:
        try:
            s3_client.head_bucket(Bucket=bucket)
            s3_client.list_objects_v2(Bucket=bucket, MaxKeys=1)
            logger.info(f"Successfully verified read access to {bucket}")
        except Exception as e:
            raise Exception(f"Failed to access bucket {bucket}: {str(e)}")
    
    for bucket in buckets_to_check['write']:
        try:
            s3_client.head_bucket(Bucket=bucket)
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

def s3_load_data(image_bucket, text_bucket, actions_bucket, common_files):
    """Load images, texts, and labels from S3."""
    s3_client = boto3.client('s3')
    images = []
    texts = []
    labels = []

    for file in common_files:
        # Load image
        image_obj = s3_client.get_object(Bucket=image_bucket, Key=f"{file}.jpg")
        image_data = image_obj['Body'].read()
        images.append(image_data)

        # Load text
        text_obj = s3_client.get_object(Bucket=text_bucket, Key=f"{file}.txt")
        text_data = text_obj['Body'].read().decode('utf-8')
        texts.append(text_data)

        # Load actions using existing parser
        action_obj = s3_client.get_object(Bucket=actions_bucket, Key=f"{file}.txt")
        action_content = action_obj['Body'].read().decode('utf-8')
        actions = parse_action_file(action_content)
        labels.append(actions)

    return images, texts, labels 