import os
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from oauth2client.service_account import ServiceAccountCredentials

def authenticate_gdrive():
    """Authenticate and return a Google Drive client."""
    scope = ["https://www.googleapis.com/auth/drive"]
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set or is empty.")
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    drive_service = build('drive', 'v3', credentials=creds)
    return drive_service

def list_jpg_files(drive_service, folder_id):
    """List all JPEG files in a specified Google Drive directory."""
    query = f"'{folder_id}' in parents and mimeType='image/jpeg'"
    results = drive_service.files().list(q=query).execute()
    return results.get('files', [])

def create_folder(drive_service, folder_name, parent_folder_id):
    """Create a folder in Google Drive."""
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_folder_id]
    }
    folder = drive_service.files().create(body=file_metadata, fields='id').execute()
    return folder.get('id')

def upload_text_content(service, folder_id, filename, content):
    """Upload text content directly to Google Drive without saving locally first."""
    file_metadata = {
        'name': filename,
        'parents': [folder_id]
    }
    
    media = MediaIoBaseUpload(
        io.BytesIO(content.encode('utf-8')),
        mimetype='text/plain',
        resumable=True
    )
    
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    
    return file.get('id')