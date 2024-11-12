import os
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from oauth2client.service_account import ServiceAccountCredentials

def authenticate_gdrive():
    """Authenticate and return a Google Drive client."""
    scope = ["https://www.googleapis.com/auth/drive"]
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    drive_service = build('drive', 'v3', credentials=creds)
    return drive_service

def list_jpg_files(drive_service, folder_id):
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