import os
from dotenv import load_dotenv
from gdrive_utils import authenticate_gdrive, list_jpg_files, download_file, upload_file, create_folder
from openai_utils import generate_frame_description, generate_action_description
from file_utils import save_text_to_file, clean_up_files

load_dotenv()  # Load environment variables from .env file

if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")

def ensure_gdrive_directories_exist(drive_service, root_folder_id, *directories):
    """Ensure that the specified directories exist in Google Drive, creating them if necessary."""
    folder_ids = {}
    
    for directory in directories:
        query = f"name='{directory}' and '{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        existing_folders = results.get('files', [])
        
        if existing_folders:
            folder_id = existing_folders[0]['id']
            print(f"Found existing directory: {directory} with ID: {folder_id}")
        else:
            folder_id = create_folder(drive_service, directory, root_folder_id)
            print(f"Created new directory: {directory} with ID: {folder_id}")
            
        folder_ids[directory] = folder_id
    
    return folder_ids

def process_frames(frames_folder_id, analysis_folder_id, actions_folder_id, system_prompt_path):
    """Process JPG frames in a Google Drive directory and save descriptions."""
    
    drive_service = authenticate_gdrive()
    jpg_files = list_jpg_files(drive_service, frames_folder_id)
    
    # Filter and sort the files to ensure they are processed in order
    jpg_files = [f for f in jpg_files if f['name'].startswith('frame_')]
    jpg_files.sort(key=lambda x: int(x['name'].split('_')[1].split('.')[0]))
    
    if len(jpg_files) < 2:
        print("Not enough frames to process.")
        return
    
    print(f"Processing {len(jpg_files)} frames...")

    for i in range(len(jpg_files) - 1):
        file_id1 = jpg_files[i]['id']
        file_name1 = jpg_files[i]['name']
        file_id2 = jpg_files[i + 1]['id']
        file_name2 = jpg_files[i + 1]['name']
        
        print(f"Downloading frames: {file_name1} and {file_name2}")
        download_file(drive_service, file_id1, file_name1)
        download_file(drive_service, file_id2, file_name2)
        
        print(f"Generating descriptions for frames: {file_name1} and {file_name2}")
        description1 = generate_frame_description(file_name1, system_prompt_path)
        description2 = generate_frame_description(file_name2, system_prompt_path)
        
        text_file_name1 = f"{os.path.splitext(file_name1)[0]}.txt"
        text_file_name2 = f"{os.path.splitext(file_name2)[0]}.txt"
        save_text_to_file(text_file_name1, description1)
        save_text_to_file(text_file_name2, description2)
        
        print(f"Uploading descriptions to frame_analysis folder")
        upload_file(drive_service, analysis_folder_id, text_file_name1)
        upload_file(drive_service, analysis_folder_id, text_file_name2)
        
        print(f"Generating action description for frames: {file_name1} and {file_name2}")
        action_description = generate_action_description(file_name1, file_name2, system_prompt_path)
        
        action_text_file_name = f"action_{os.path.splitext(file_name1)[0]}_{os.path.splitext(file_name2)[0]}.txt"
        save_text_to_file(action_text_file_name, action_description)
        
        print(f"Uploading action description to actions_analysis folder")
        upload_file(drive_service, actions_folder_id, action_text_file_name)
        
        print(f"Cleaning up local files for frames: {file_name1} and {file_name2}")
        clean_up_files(file_name1, file_name2, text_file_name1, text_file_name2, action_text_file_name)

    print("Game analysis complete.")

if __name__ == "__main__":
    root_folder_id = input("Please enter root Google Drive folder ID: ")
    drive_service = authenticate_gdrive()
    folder_ids = ensure_gdrive_directories_exist(drive_service, root_folder_id, 'frame_analysis', 'actions_analysis', 'frames')
    frames_folder_id = folder_ids['frames']
    analysis_folder_id = folder_ids['frame_analysis']
    actions_folder_id = folder_ids['actions_analysis']
    system_prompt_path = 'system_prompt.txt'  # Path to the system prompt file
    process_frames(frames_folder_id, analysis_folder_id, actions_folder_id, system_prompt_path)