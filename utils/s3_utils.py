import boto3
import logging
from botocore.exceptions import ClientError
from pathlib import Path
import json
from utils.actions import parse_action_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def s3_verify_bucket_access():
    """Verify access to required S3 buckets"""
    # Keep this for pre-training checks
    ...

def s3_upload_file(file_path, bucket, key):
    """Upload a file to S3"""
    # Useful for model deployment and other operations
    ...

def s3_download_file(bucket, key, file_path):
    """Download a file from S3"""
    # Useful for inference and testing
    ...