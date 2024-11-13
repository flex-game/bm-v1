import os
import csv
from gdrive_utils import authenticate_gdrive, list_txt_files, download_file_content, upload_text_content
import logging
from typing import List, Dict
from google.api_core import retry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@retry.Retry()
def extract_txt_files(drive_service, root_folder_id: str, target_folder_name: str) -> List[Dict]:
    """
    Extract all .txt files from frame_analysis folders in all training subdirectories.
    
    Args:
        drive_service: Authenticated Google Drive service instance
        root_folder_id: ID of the root folder (bm-v1)
        target_folder_name: Name of target folder (frame_analysis)
    
    Returns:
        List of dictionaries containing file metadata
    
    Raises:
        FileNotFoundError: If training_data folder is not found
        Exception: For other API or processing errors

    """

    all_txt_files = []
    
    try:
        # Find training_data folder
        query = f"'{root_folder_id}' in parents and name='training_data' and mimeType='application/vnd.google-apps.folder'"
        results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        training_folders = results.get('files', [])
        
        if not training_folders:
            raise FileNotFoundError("training_data folder not found in root directory")
        
        training_folder = training_folders[0]
        logger.info(f"Found training_data folder: {training_folder['name']}")
        
        # Get all training subfolders
        query = f"'{training_folder['id']}' in parents and mimeType='application/vnd.google-apps.folder'"
        results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        training_subfolders = results.get('files', [])
        
        if not training_subfolders:
            logger.warning("No training subfolders found in training_data")
            return all_txt_files
            
        logger.info(f"Found {len(training_subfolders)} training subfolders")
        
        # Process each training subfolder
        for subfolder in training_subfolders:
            try:
                logger.info(f"Processing subfolder: {subfolder['name']}")
                
                # Find frame_analysis folder
                query = f"'{subfolder['id']}' in parents and name='{target_folder_name}' and mimeType='application/vnd.google-apps.folder'"
                results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
                frame_analysis_folders = results.get('files', [])
                
                if not frame_analysis_folders:
                    logger.warning(f"No {target_folder_name} folder found in {subfolder['name']}")
                    continue
                
                # Get txt files from frame_analysis
                frame_analysis_id = frame_analysis_folders[0]['id']
                txt_files = list_txt_files(drive_service, frame_analysis_id)
                
                if txt_files:
                    all_txt_files.extend(txt_files)
                    logger.info(f"Added {len(txt_files)} txt files from {subfolder['name']}/{target_folder_name}")
                else:
                    logger.warning(f"No txt files found in {subfolder['name']}/{target_folder_name}")
                
            except Exception as e:
                logger.error(f"Error processing subfolder {subfolder['name']}: {str(e)}")
                continue
        
        logger.info(f"Total txt files collected: {len(all_txt_files)}")
        return all_txt_files
        
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in extract_txt_files: {str(e)}")
        raise

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
    root_folder_id = '14rV_AfSINfFyUQgZN4wJEgGtJCyzlv0a'
    drive_service = authenticate_gdrive()
    
    # Extract .txt files from frame_analysis and actions_analysis subdirectories
    frame_txt_files = extract_txt_files(drive_service, root_folder_id, 'frame_analysis')
    action_txt_files = extract_txt_files(drive_service, root_folder_id, 'actions_analysis')
    
    # Compile and upload CSV
    compile_to_csv(drive_service, root_folder_id, frame_txt_files, action_txt_files)
    print("Dataset compilation complete and uploaded as dataset.csv.")

if __name__ == "__main__":
    main() 