import boto3
import os
import logging
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_s3_client():
    try:
        # Create a session using AWS credentials from environment variables or IAM roles
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        logging.info("S3 client session created successfully.")
        return session.client('s3')
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Error: {e}")
        return None

def create_bucket(s3, bucket_name):
    try:
        s3.create_bucket(Bucket=bucket_name)
        logging.info(f"Bucket '{bucket_name}' created successfully.")
    except ClientError as e:
        logging.error(f"Failed to create bucket '{bucket_name}': {e}")

def upload_files_to_s3(s3, bucket_name, local_directory):
    if not s3:
        logging.error("S3 client could not be initialized. Check your AWS credentials.")
        return

    for root, dirs, files in os.walk(local_directory):
        for file in files:
            if file.endswith('.json'):
                local_file_path = os.path.join(root, file)
                s3_key = file  # You can modify this if you want a different key structure in S3
                try:
                    s3.upload_file(local_file_path, bucket_name, s3_key)
                    logging.info(f"Uploaded '{local_file_path}' to bucket '{bucket_name}' as '{s3_key}'.")
                except Exception as e:
                    logging.error(f"Failed to upload '{local_file_path}' to S3: {e}")

def main():
    bucket_name = 'bm-v1-training-text-json'
    local_directory = 'local/json_files/'

    s3 = get_s3_client()

    if s3:
        create_bucket(s3, bucket_name)
        upload_files_to_s3(s3, bucket_name, local_directory)

if __name__ == "__main__":
    main() 