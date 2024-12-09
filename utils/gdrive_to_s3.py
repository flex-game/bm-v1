import os
import logging
import boto3
from gdrive_utils import authenticate_gdrive, list_jpg_files, download_file_content
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_to_s3(s3_client, bucket_name, file_name, file_content):
    """Upload a file to an S3 bucket."""
    try:
        s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=file_content)
        logger.info(f"Uploaded {file_name} to S3 bucket {bucket_name}")
    except Exception as e:
        logger.error(f"Failed to upload {file_name} to S3: {str(e)}")

def process_gdrive_to_s3(drive_service, s3_client, root_folder_id, bucket_name):
    """Process Google Drive folders and upload images to S3."""
    query = f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
    results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    subfolders = results.get('files', [])

    for subfolder in subfolders:
        subfolder_id = subfolder['id']
        query = f"'{subfolder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
        results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        folders = {folder['name']: folder['id'] for folder in results.get('files', [])}

        frames_folder_id = folders.get('frames')
        if frames_folder_id:
            jpg_files = list_jpg_files(drive_service, frames_folder_id)
            for file in jpg_files:
                try:
                    file_id = file['id']
                    file_name = file['name']
                    parent_folder_name = subfolder['name']
                    new_file_name = f"{parent_folder_name}_{file_name}"

                    # Download the file content
                    file_content = download_file_content(drive_service, file_id)

                    # Upload to S3
                    upload_to_s3(s3_client, bucket_name, new_file_name, file_content)
                except Exception as e:
                    logger.error(f"Error processing file {file['name']}: {str(e)}")
        else:
            logger.warning(f"No frames folder found in subfolder {subfolder['name']}")

def main():
    # Authenticate with Google Drive
    drive_service = authenticate_gdrive()

    # Authenticate with AWS S3
    s3_client = boto3.client('s3')

    # Define the root folder ID and S3 bucket name
    root_folder_id = '1BdyuWOoHuoeirHS7GwuMe77V3Cd35i_m'  # Replace with your actual root folder ID
    bucket_name = 'buddy_model_v1_dataset'

    # Create the S3 bucket if it doesn't exist
    try:
        s3_client.create_bucket(Bucket=bucket_name)
        logger.info(f"Created S3 bucket: {bucket_name}")
    except s3_client.exceptions.BucketAlreadyExists:
        logger.info(f"S3 bucket {bucket_name} already exists.")
    except s3_client.exceptions.BucketAlreadyOwnedByYou:
        logger.info(f"S3 bucket {bucket_name} already owned by you.")

    # Process Google Drive folders and upload images to S3
    process_gdrive_to_s3(drive_service, s3_client, root_folder_id, bucket_name)

if __name__ == "__main__":
    main() 