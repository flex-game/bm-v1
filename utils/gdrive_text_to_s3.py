import os
import logging
import boto3
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def authenticate_gdrive():
    """Authenticate with Google Drive API using service account credentials."""
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not found in .env file")
    
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    return build('drive', 'v3', credentials=credentials)

def list_txt_files(service, folder_id):
    """List all text files in a Google Drive folder."""
    query = f"'{folder_id}' in parents and mimeType='text/plain' and trashed=false"
    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)'
    ).execute()
    return results.get('files', [])

def download_file_content(service, file_id):
    """Download a file's content from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    file_content = io.BytesIO()
    downloader = MediaIoBaseDownload(file_content, request)
    done = False
    while done is False:
        _, done = downloader.next_chunk()
    return file_content.getvalue()

def upload_to_s3(s3_client, bucket_name, file_name, file_content):
    """Upload a file to an S3 bucket."""
    try:
        s3_client.put_object(
            Bucket=bucket_name, 
            Key=file_name, 
            Body=file_content
        )
        logger.info(f"Uploaded {file_name} to S3 bucket {bucket_name}")
    except Exception as e:
        logger.error(f"Failed to upload {file_name} to S3: {str(e)}")

def process_gdrive_to_s3(drive_service, s3_client, root_folder_id, bucket_name):
    """Process Google Drive folders and upload text files to S3."""
    query = f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
    results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    subfolders = results.get('files', [])

    for subfolder in subfolders:
        subfolder_id = subfolder['id']
        query = f"'{subfolder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
        results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        folders = {folder['name']: folder['id'] for folder in results.get('files', [])}

        frame_analysis_folder_id = folders.get('frame_analysis')
        if frame_analysis_folder_id:
            txt_files = list_txt_files(drive_service, frame_analysis_folder_id)
            for file in txt_files:
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
            logger.warning(f"No frame_analysis folder found in subfolder {subfolder['name']}")

def main():
    # Authenticate with Google Drive
    drive_service = authenticate_gdrive()

    # Initialize AWS S3 client using default credentials
    s3_client = boto3.client('s3')

    # Define the root folder ID and S3 bucket name
    root_folder_id = '1BdyuWOoHuoeirHS7GwuMe77V3Cd35i_m'
    bucket_name = 'bm-v1-training-text'

    # Process Google Drive folders and upload text files to S3
    process_gdrive_to_s3(drive_service, s3_client, root_folder_id, bucket_name)

if __name__ == "__main__":
    main() 