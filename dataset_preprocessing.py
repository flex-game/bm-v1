import os
import csv
import json
import logging
from gdrive_utils import authenticate_gdrive, list_txt_files, list_jpg_files, download_file_content
import pandas as pd
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaIoBaseDownload
import io

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def collect_image_paths(drive_service, frames_folder_id):
    """Collect Google Drive URLs for images in the frames directory."""
    jpg_files = list_jpg_files(drive_service, frames_folder_id)
    
    # Create direct access URLs for each image
    image_paths = []
    for file in jpg_files:
        # Create a shareable link using the file ID
        file_id = file['id']
        # Direct download URL format for Google Drive
        url = f"https://drive.google.com/uc?id={file_id}"
        image_paths.append(url)
    
    logger.info(f"Collected {len(image_paths)} image URLs from frames folder.")
    return image_paths

def collect_stats_shot(drive_service, frame_analysis_folder_id, subfolder_name):
    """Collect stats-shot data from frame_analysis folders."""
    txt_files = list_txt_files(drive_service, frame_analysis_folder_id)
    stats_shots = []
    
    for file in txt_files:
        try:
            content = download_file_content(drive_service, file['id'])
            
            # Clean the content
            content = content.strip()
            if not content:
                logger.warning(f"Empty content in file: {file['name']}")
                continue
                
            # Remove any markdown code block markers if present
            if content.startswith('```') and content.endswith('```'):
                content = content[content.find('{'):content.rfind('}')+1]
            
            logger.debug(f"Processing content from {file['name']}: {content[:100]}...")
            stats_shot = json.loads(content)
            
            # Validate the parsed JSON has expected fields
            if not isinstance(stats_shot, dict):
                logger.warning(f"Unexpected JSON structure in {file['name']}: not a dictionary")
                continue
                
            # Add metadata
            stats_shot['subfolder'] = subfolder_name
            stats_shot['filename'] = file['name']
            
            stats_shots.append(stats_shot)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in file {file['name']}: {str(e)}")
            logger.error(f"Raw content: {content}")
            continue
        except Exception as e:
            logger.error(f"Error processing file {file['name']}: {str(e)}")
            continue
    
    logger.info(f"Collected {len(stats_shots)} valid stats-shot entries from {subfolder_name}")
    return stats_shots

def compile_unique_actions(drive_service, root_folder_id):
    """Compile a set of all unique actions from all actions_analysis folders."""
    all_actions = set()
    query = f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
    results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    subfolders = results.get('files', [])
    
    for subfolder in subfolders:
        subfolder_id = subfolder['id']
        query = f"'{subfolder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
        results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        folders = {folder['name']: folder['id'] for folder in results.get('files', [])}
        
        actions_analysis_folder_id = folders.get('actions_analysis')
        if actions_analysis_folder_id:
            txt_files = list_txt_files(drive_service, actions_analysis_folder_id)
            for file in txt_files:
                try:
                    content = download_file_content(drive_service, file['id'])
                    # Clean the content by removing whitespace and newlines
                    content = content.strip()
                    
                    # Extract the list part between square brackets
                    start_idx = content.find('[')
                    end_idx = content.rfind(']')
                    if start_idx != -1 and end_idx != -1:
                        actions_list = content[start_idx + 1:end_idx]
                        # Split by commas and clean up each action
                        actions = [action.strip().strip('"') for action in actions_list.split(',') if action.strip()]
                        all_actions.update(actions)
                    else:
                        logger.warning(f"Could not find action list in file {file['name']}")
                except Exception as e:
                    logger.error(f"Error processing file {file['name']}: {str(e)}")
                    continue
    
    # Export actions list to CSV
    actions_df = pd.DataFrame({'action': list(all_actions)})
    actions_df.to_csv('unique_actions.csv', index=False)
    logger.info(f"Exported {len(all_actions)} unique actions to unique_actions.csv")
    
    return list(all_actions)

def compile_action_labels(file_path):
    logging.info(f"Reading actions from CSV: {file_path}")
    try:
        # Read from CSV instead of JSON
        actions_data = []
        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            # Skip header if it exists
            next(csv_reader, None)
            actions_data = [row[0] for row in csv_reader]  # Assuming actions are in first column
        return actions_data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return []

def check_file_in_drive(drive_service, folder_id, filename):
    """Check if file exists in Google Drive folder and return file ID if found."""
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields='files(id, name)').execute()
    files = results.get('files', [])
    
    if files:
        logger.info(f"Found existing {filename} in Drive")
        return files[0]['id']
    return None

def upload_to_drive(drive_service, file_path, folder_id, file_name=None):
    """Upload a file to Google Drive, replacing if it already exists."""
    if file_name is None:
        file_name = os.path.basename(file_path)
    
    # Check if file already exists
    existing_file_id = check_file_in_drive(drive_service, folder_id, file_name)
    
    media = MediaFileUpload(
        file_path,
        mimetype='text/csv',
        resumable=True
    )
    
    if existing_file_id:
        # Update existing file
        file = drive_service.files().update(
            fileId=existing_file_id,
            media_body=media,
            fields='id'
        ).execute()
        logger.info(f"Updated existing {file_name} in Drive with ID: {file.get('id')}")
    else:
        # Create new file
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        logger.info(f"Uploaded new {file_name} to Drive with ID: {file.get('id')}")
    
    return file.get('id')

def download_file_from_drive(drive_service, file_id, filename):
    """Download a file from Google Drive."""
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.seek(0)
    
    with open(filename, 'wb') as f:
        f.write(fh.read())
    logger.info(f"Downloaded {filename} from Drive")

def main():
    root_folder_id = '1BdyuWOoHuoeirHS7GwuMe77V3Cd35i_m'
    drive_service = authenticate_gdrive()
    
    logger.info("Starting new data processing pipeline...")
    
    # Check for existing files in Drive first, then locally
    proceed_with_actions = True
    proceed_with_stats = True
    
    # Check for unique_actions.csv
    actions_file_id = check_file_in_drive(drive_service, root_folder_id, 'unique_actions.csv')
    if actions_file_id or os.path.exists('unique_actions.csv'):
        logger.info("Found existing unique_actions.csv, loading actions...")
        try:
            # Download the file if it exists in Drive but not locally
            if actions_file_id and not os.path.exists('unique_actions.csv'):
                download_file_from_drive(drive_service, actions_file_id, 'unique_actions.csv')
            
            actions_df = pd.read_csv('unique_actions.csv')
            all_actions = actions_df['action'].tolist()
            logger.info(f"Loaded {len(all_actions)} actions from existing file")
            proceed_with_actions = False
        except Exception as e:
            logger.error(f"Error loading unique_actions.csv: {e}")
            logger.info("Falling back to collecting actions from Drive...")
            proceed_with_actions = True
    
    if proceed_with_actions:
        logger.info("Collecting actions from Drive...")
        all_actions = compile_unique_actions(drive_service, root_folder_id)
    
    # Check for all_stats_shots.csv
    stats_file_id = check_file_in_drive(drive_service, root_folder_id, 'all_stats_shots.csv')
    if stats_file_id or os.path.exists('all_stats_shots.csv'):
        logger.info("Found existing all_stats_shots.csv, loading stats...")
        try:
            # Download the file if it exists in Drive but not locally
            if stats_file_id and not os.path.exists('all_stats_shots.csv'):
                download_file_from_drive(drive_service, stats_file_id, 'all_stats_shots.csv')
            
            stats_df = pd.read_csv('all_stats_shots.csv')
            all_stats_shots = stats_df.to_dict('records')
            logger.info(f"Loaded {len(all_stats_shots)} stats shots from existing file")
            proceed_with_stats = False
        except Exception as e:
            logger.error(f"Error loading all_stats_shots.csv: {e}")
            logger.info("Falling back to collecting stats shots from Drive...")
            proceed_with_stats = True
    
    if proceed_with_stats:
        logger.info("Collecting stats shots from Drive...")
        # ... rest of the collection code ...
    
    # Add this section to collect image paths
    logger.info("Collecting image paths...")
    query = f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
    results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    subfolders = results.get('files', [])
    
    image_paths = []
    for subfolder in subfolders:
        subfolder_id = subfolder['id']
        query = f"'{subfolder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
        results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        folders = {folder['name']: folder['id'] for folder in results.get('files', [])}
        
        frames_folder_id = folders.get('frames')
        if frames_folder_id:
            subfolder_images = collect_image_paths(drive_service, frames_folder_id)
            image_paths.extend(subfolder_images)
            logger.info(f"Added {len(subfolder_images)} images from subfolder {subfolder['name']}")
        else:
            logger.warning(f"Could not find frames folder in subfolder {subfolder['name']}")

    logger.info(f"Total images collected: {len(image_paths)}")

    # Before exporting image paths
    image_paths_file = 'image_paths.csv'
    
    # Check if file already exists in Drive
    try:
        file_id = check_file_in_drive(drive_service, root_folder_id, image_paths_file)
        if file_id:
            logging.info(f"{image_paths_file} already exists in Drive, downloading...")
            # Download the file first
            download_file_from_drive(drive_service, file_id, image_paths_file)
            logging.info("Using existing image paths file")
            with open(image_paths_file, 'r') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)  # Skip header
                image_paths = [row[0] for row in csv_reader]
        else:
            logging.info(f"Exporting {len(image_paths)} image paths to CSV...")
            with open(image_paths_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_url'])  # Changed header to reflect URL nature
                for url in image_paths:
                    writer.writerow([url])
            
            # Upload to Google Drive
            upload_to_drive(drive_service, image_paths_file, root_folder_id, 'image_paths.csv')
            logging.info(f"Successfully uploaded {image_paths_file} to Google Drive")
    except Exception as e:
        logging.error(f"Error handling image paths file: {str(e)}")
        raise

    # Continue with dataset.csv creation
    logging.info("Creating final dataset.csv...")
    
    # Load the data from CSV files
    stats_df = pd.read_csv('all_stats_shots.csv')
    image_paths_df = pd.read_csv('image_paths.csv')
    actions_df = pd.read_csv('unique_actions.csv')
    
    with open('dataset.csv', mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Header row: image_path, stats_shot, followed by all action columns
        writer.writerow(['image_path', 'stats_shot'] + actions_df['action'].tolist())
        
        # Write data rows
        for i in range(len(image_paths_df)):
            image_path = image_paths_df.iloc[i]['image_url']
            stats_shot = stats_df.iloc[i].to_dict() if i < len(stats_df) else {}
            # Initialize all actions as 0 (assuming binary labels)
            action_values = [0] * len(actions_df)
            writer.writerow([image_path, json.dumps(stats_shot)] + action_values)
    
    # Upload files to Drive only after all files are created
    logger.info("Uploading files to Google Drive...")
    for file_name in ['unique_actions.csv', 'all_stats_shots.csv', 'dataset.csv']:
        if os.path.exists(file_name):
            upload_to_drive(drive_service, file_name, root_folder_id)
        else:
            logger.warning(f"File {file_name} not found, skipping upload")
    
    # Clean up local files
    logger.info("Cleaning up local files...")
    for file in ['unique_actions.csv', 'all_stats_shots.csv', 'dataset.csv', 'image_paths.csv']:
        try:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Removed {file}")
        except Exception as e:
            logger.warning(f"Could not remove {file}: {str(e)}")

    logger.info("Dataset compilation complete!")

if __name__ == "__main__":
    main() 