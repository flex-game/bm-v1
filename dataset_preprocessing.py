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
def extract_txt_files(drive_service, root_folder_id: str, target_folder_name: str) -> Dict[str, List[Dict]]:
    """
    Extract all .txt files from frame_analysis folders in all training subdirectories.
    
    Args:
        drive_service: Authenticated Google Drive service instance
        root_folder_id: ID of the root folder (bm-v1)
        target_folder_name: Name of target folder (frame_analysis)
    
    Returns:
        Dictionary with subfolder names as keys and lists of file metadata as values
    
    Raises:
        FileNotFoundError: If training_data folder is not found
        Exception: For other API or processing errors

    """

    all_txt_files = {}
    
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
                    all_txt_files[subfolder['name']] = txt_files
                    logger.info(f"Added {len(txt_files)} txt files from {subfolder['name']}/{target_folder_name}")
                else:
                    logger.warning(f"No txt files found in {subfolder['name']}/{target_folder_name}")
                
            except Exception as e:
                logger.error(f"Error processing subfolder {subfolder['name']}: {str(e)}")
                continue
        
        logger.info(f"Total subfolders with txt files collected: {len(all_txt_files)}")
        return all_txt_files
        
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in extract_txt_files: {str(e)}")
        raise

def sanitize_text_for_csv(text: str) -> str:
    """
    Sanitize text for CSV export by removing or replacing problematic characters.

    Args:
        text: The text content to sanitize.

    Returns:
        A sanitized version of the text.
    """
    # Replace commas with a space or another character to avoid CSV column issues
    sanitized_text = text.replace(',', ' ')
    # Optionally, replace newlines with a space or another character
    sanitized_text = sanitized_text.replace('\n', ' ').replace('\r', ' ')
    # Handle quotes by escaping them or removing them
    sanitized_text = sanitized_text.replace('"', '""')
    return sanitized_text

def compile_to_csv(drive_service, root_folder_id, frame_txt_files, action_txt_files):
    """Compile .txt files into a single CSV, linking frame and action analysis files."""
    
    csv_content = []
    
    # Iterate over the keys in the frame_txt_files dictionary
    for root_folder in frame_txt_files.keys():
        frame_files = frame_txt_files.get(root_folder, [])
        action_files = action_txt_files.get(root_folder, [])
        
        # Ensure both frame and action files exist for the root folder
        if frame_files and action_files:
            logger.info(f"Processing files from root folder: {root_folder}")
            
            # Assuming each root folder has a list of files, process them
            for frame_file, action_file in zip(frame_files, action_files):
                frame_content = download_file_content(drive_service, frame_file['id'])
                action_content = download_file_content(drive_service, action_file['id'])
                
                # Sanitize the content before adding to CSV
                sanitized_frame_content = sanitize_text_for_csv(frame_content)
                sanitized_action_content = sanitize_text_for_csv(action_content)
                
                csv_content.append([sanitized_frame_content, sanitized_action_content])
                logger.info(f"Added files from {root_folder} to CSV content")
    
    # Write all content to a single CSV
    csv_file_path = 'dataset.csv'
    logger.info(f"Writing all data to {csv_file_path}")
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Frame Analysis', 'Action Analysis'])  # Add headers
        writer.writerows(csv_content)
    
    # Upload CSV to root folder
    logger.info(f"Uploading {csv_file_path} to Google Drive")
    upload_text_content(drive_service, root_folder_id, csv_file_path, open(csv_file_path, 'r').read())
    os.remove(csv_file_path)  # Clean up local file
    logger.info(f"CSV file {csv_file_path} uploaded and local copy removed")

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