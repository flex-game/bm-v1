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

def list_txt_files(drive_service, folder_id: str):
    """
    List all .txt files in a specified Google Drive folder.

    Args:
        drive_service: Authenticated Google Drive service instance
        folder_id: ID of the folder to search for .txt files

    Returns:
        List of dictionaries containing file metadata
    """
    query = f"'{folder_id}' in parents and mimeType='text/plain'"
    results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    txt_files = results.get('files', [])
    return txt_files

def download_file_content(drive_service, file_id: str) -> str:
    """
    Download the content of a file from Google Drive.

    Args:
        drive_service: Authenticated Google Drive service instance
        file_id: ID of the file to download

    Returns:
        The content of the file as a string
    """
    request = drive_service.files().get_media(fileId=file_id)
    file_content = io.BytesIO()
    downloader = MediaIoBaseDownload(file_content, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    file_content.seek(0)
    return file_content.read().decode('utf-8')