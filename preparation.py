import os
import io
import gspread
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import openai

load_dotenv()  # Load environment variables from .env file

def authenticate_gdrive():
    """Authenticate and return a Google Drive client."""
    scope = ["https://www.googleapis.com/auth/drive"]
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    drive_service = build('drive', 'v3', credentials=creds)
    return drive_service

def list_jpg_files_in_directory(drive_service, folder_id):
    """List all JPEG files in a specified Google Drive directory."""
    query = f"'{folder_id}' in parents and mimeType='image/jpeg'"
    results = drive_service.files().list(q=query).execute()
    return results.get('files', [])

def download_file(drive_service, file_id, file_name):
    """Download a file from Google Drive."""
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.seek(0)
    with open(file_name, 'wb') as f:
        f.write(fh.read())

def upload_file(drive_service, folder_id, file_name, mime_type='text/plain'):
    """Upload a file to Google Drive."""
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_name, mimetype=mime_type)
    drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

def generate_image_description(image_path):
    """Generate a description for an image using OpenAI."""
    openai.api_key = os.getenv('OPENAI_API_KEY')
    with open(image_path, 'rb') as image_file:
        response = openai.Image.create(file=image_file, purpose='description')
    return response['choices'][0]['text']

def process_images_in_directory(folder_id):
    """Process JPEG images in a Google Drive directory and save descriptions."""
    drive_service = authenticate_gdrive()
    jpg_files = list_jpg_files_in_directory(drive_service, folder_id)
    
    for jpg_file in jpg_files:
        file_id = jpg_file['id']
        file_name = jpg_file['name']
        
        # Download the image
        download_file(drive_service, file_id, file_name)
        
        # Generate description
        description = generate_image_description(file_name)
        
        # Save description to a text file
        text_file_name = f"{os.path.splitext(file_name)[0]}.txt"
        with open(text_file_name, 'w') as text_file:
            text_file.write(description)
        
        # Upload the text file back to Google Drive
        upload_file(drive_service, folder_id, text_file_name)
        
        # Clean up local files
        os.remove(file_name)
        os.remove(text_file_name)

if __name__ == "__main__":
    folder_id = 'YOUR_GOOGLE_DRIVE_FOLDER_ID'  # Update with your actual folder ID
    process_images_in_directory(folder_id) 