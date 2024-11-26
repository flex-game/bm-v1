import os
import csv
import json
import logging
from gdrive_utils import authenticate_gdrive, list_txt_files, list_jpg_files, download_file_content
import pandas as pd
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaIoBaseDownload
import io
import sys

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

def parse_action_file_content(content):
    """
    Parse the action file content, handling both JSON and direct text formats
    Returns the list of actions or None if parsing fails
    """
    try:
        # First attempt: try to parse as JSON
        data = json.loads(content)
        if 'actions_by_player' in data:
            return data['actions_by_player']
    except json.JSONDecodeError:
        # Second attempt: try to parse the content directly
        try:
            # Look for the actions_by_player list using string manipulation
            start_marker = '"actions_by_player": ['
            end_marker = ']'
            
            start_idx = content.find(start_marker)
            if start_idx == -1:
                return None
                
            start_idx += len(start_marker)
            end_idx = content.find(end_marker, start_idx)
            
            if end_idx == -1:
                return None
                
            actions_str = content[start_idx:end_idx]
            # Split by commas and clean up the strings
            actions = [
                action.strip().strip('"') 
                for action in actions_str.split('",')
                if action.strip()
            ]
            return actions
            
        except Exception as e:
            logging.warning(f"Failed to parse file content directly: {e}")
            return None
    
    return None

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
    
    # Check for existing image_urls.csv first
    image_urls_file = 'image_urls.csv'
    image_paths = []
    
    # Check for file in Drive and locally
    file_id = check_file_in_drive(drive_service, root_folder_id, image_urls_file)
    if file_id or os.path.exists(image_urls_file):
        logger.info("Found existing image_urls.csv, loading paths...")
        try:
            # Download the file if it exists in Drive but not locally
            if file_id and not os.path.exists(image_urls_file):
                download_file_from_drive(drive_service, file_id, image_urls_file)
            
            # Read the existing paths
            with open(image_urls_file, 'r') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)  # Skip header
                image_paths = [row[0] for row in csv_reader]
            logger.info(f"Loaded {len(image_paths)} image paths from existing file")
        except Exception as e:
            logger.error(f"Error loading image_urls.csv: {e}")
            logger.info("Falling back to collecting image paths from Drive...")
            image_paths = []
    
    # Only collect paths if we don't have them already
    if not image_paths:
        query = f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
        results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        subfolders = results.get('files', [])
        
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

        # Save the newly collected paths
        with open(image_urls_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_url'])
            for url in image_paths:
                writer.writerow([url])
        
        # Upload to Google Drive
        upload_to_drive(drive_service, image_urls_file, root_folder_id)
        logger.info(f"Successfully uploaded {image_urls_file} to Google Drive")

    # Continue with dataset.csv creation
    logging.info("Creating final dataset.csv...")
    
    # Load the data from CSV files
    stats_df = pd.read_csv('all_stats_shots.csv')
    image_paths_df = pd.read_csv('image_urls.csv')
    actions_df = pd.read_csv('unique_actions.csv')
    all_actions = actions_df['action'].tolist()
    
    with open('dataset.csv', mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Header row: image_path, stats_shot, followed by all action columns
        writer.writerow(['image_path', 'stats_shot'] + all_actions)
        
        # Write data rows
        for i in range(len(image_paths_df)):
            image_path = image_paths_df.iloc[i]['image_url']
            stats_shot = stats_df.iloc[i].to_dict() if i < len(stats_df) else {}
            
            # Get the subfolder name from the stats_shot
            subfolder = stats_shot.get('subfolder', '')
            
            # Initialize all actions as 0
            action_values = [0] * len(all_actions)
            
            try:
                # Get the file name from the Drive file ID
                file_id = image_path.split('id=')[1]
                file_metadata = drive_service.files().get(fileId=file_id, fields='name').execute()
                image_number = file_metadata['name'].split('.')[0]  # Get number from actual filename
                
                # Get subfolder ID
                query = f"name='{subfolder}' and '{root_folder_id}' in parents"
                results = drive_service.files().list(q=query, fields='files(id)').execute()
                subfolder_files = results.get('files', [])
                
                if subfolder_files:
                    subfolder_id = subfolder_files[0]['id']
                    
                    # Get actions_analysis folder ID
                    query = f"name='actions_analysis' and '{subfolder_id}' in parents"
                    results = drive_service.files().list(q=query, fields='files(id)').execute()
                    actions_folder = results.get('files', [])
                    
                    if actions_folder:
                        actions_folder_id = actions_folder[0]['id']
                        
                        # Look for action files with the correct image number
                        query = f"name contains 'action_{image_number}_' and mimeType='text/plain' and '{actions_folder_id}' in parents"
                        results = drive_service.files().list(q=query, fields='files(id)').execute()
                        action_files = results.get('files', [])
                        
                        for file_info in action_files:
                            try:
                                # Add debug logging to see what we're getting
                                logging.debug(f"File info received: {file_info}")
                                
                                if not isinstance(file_info, dict):
                                    logging.error(f"file_info is not a dictionary: {type(file_info)}")
                                    continue
                                    
                                if 'name' not in file_info:
                                    logging.error(f"'name' not found in file_info keys: {file_info.keys()}")
                                    continue
                                
                                filename = file_info['name']
                                logging.info(f"Processing file: {filename}")
                                
                                # Extract the base action number(s) from filename
                                # This will handle both 'action_35' and 'action_35_36' formats
                                action_nums = filename.replace('action_', '').split('_')
                                logging.debug(f"Extracted action numbers: {action_nums}")
                                
                                content = download_file_content(drive_service, file_info['id'])
                                if not content:
                                    logging.warning(f"No content found for file {filename}")
                                    continue
                                        
                                actions = parse_action_file_content(content)
                                
                                if actions:
                                    # Process the actions, associating them with all relevant action numbers
                                    for action_num in action_nums:
                                        # Add to your processing logic here
                                        pass
                                else:
                                    logging.warning(f"No valid actions found in file {filename} ({action_nums})")
                                
                            except Exception as e:
                                logging.error(f"Error processing action file: {str(e)}")
                                logging.error(f"File info that caused error: {file_info}")
                                continue
                    else:
                        logger.warning(f"No action file found for image {image_number} in {subfolder}")
            
            except Exception as e:
                logger.warning(f"Error processing actions for image {image_path}: {str(e)}")
            
            writer.writerow([image_path, json.dumps(stats_shot)] + action_values)
            
            if i % 50 == 0:  # Log progress every 50 images
                logger.info(f"Processed {i}/{len(image_paths_df)} images")
    
    # Upload files to Drive only after all files are created
    logger.info("Uploading files to Google Drive...")
    for file_name in ['unique_actions.csv', 'all_stats_shots.csv', 'dataset.csv']:
        if os.path.exists(file_name):
            upload_to_drive(drive_service, file_name, root_folder_id)
        else:
            logger.warning(f"File {file_name} not found, skipping upload")
    
    # Clean up local files
    logger.info("Cleaning up local files...")
    for file in ['unique_actions.csv', 'all_stats_shots.csv', 'dataset.csv', 'image_urls.csv']:
        try:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Removed {file}")
        except Exception as e:
            logger.warning(f"Could not remove {file}: {str(e)}")

    logger.info("Dataset compilation complete!")

if __name__ == "__main__":
    main() 