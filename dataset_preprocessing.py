import os
import csv
from gdrive_utils import authenticate_gdrive, list_txt_files, download_file_content, upload_text_content

def extract_txt_files(drive_service, root_folder_id, subfolder_name):
    """Extract all .txt files from a specified subfolder."""
    query = f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    subfolders = results.get('files', [])
    
    txt_files = []
    for subfolder in subfolders:
        if subfolder['name'] == subfolder_name:
            subfolder_id = subfolder['id']
            txt_files.extend(list_txt_files(drive_service, subfolder_id))
    
    return txt_files

def compile_to_csv(drive_service, root_folder_id, frame_txt_files, action_txt_files):
    """Compile .txt files into a CSV and upload it to the root folder."""
    csv_content = []
    
    # Add headers
    csv_content.append(['File Name', 'Content'])
    
    # Add frame analysis files
    for file in frame_txt_files:
        file_content = download_file_content(drive_service, file['id'])
        csv_content.append([file['name'], file_content])
    
    # Add action analysis files
    for file in action_txt_files:
        file_content = download_file_content(drive_service, file['id'])
        csv_content.append([file['name'], file_content])
    
    # Write to CSV
    csv_file_path = 'dataset.csv'
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_content)
    
    # Upload CSV to root folder
    upload_text_content(drive_service, root_folder_id, 'dataset.csv', open(csv_file_path, 'r').read())
    os.remove(csv_file_path)  # Clean up local file

def main():
    root_folder_id = input("Please enter root Google Drive folder ID: ")
    drive_service = authenticate_gdrive()
    
    # Extract .txt files from frame_analysis and actions_analysis subdirectories
    frame_txt_files = extract_txt_files(drive_service, root_folder_id, 'frame_analysis')
    action_txt_files = extract_txt_files(drive_service, root_folder_id, 'actions_analysis')
    
    # Compile and upload CSV
    compile_to_csv(drive_service, root_folder_id, frame_txt_files, action_txt_files)
    print("Dataset compilation complete and uploaded as dataset.csv.")

if __name__ == "__main__":
    main() 