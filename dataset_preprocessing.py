import os
import csv
import json
import logging
from gdrive_utils import authenticate_gdrive, list_txt_files, list_jpg_files, download_file_content
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def collect_image_paths(drive_service, frames_folder_id):
    """Collect paths to images in the frames directory."""
    jpg_files = list_jpg_files(drive_service, frames_folder_id)
    image_paths = [f"frames/{file['name']}" for file in jpg_files]
    logger.info(f"Collected {len(image_paths)} image paths from frames folder.")
    return image_paths

def collect_stats_shot(drive_service, frame_analysis_folder_id, subfolder_name):
    """Collect stats-shot data from frame_analysis folders."""
    txt_files = list_txt_files(drive_service, frame_analysis_folder_id)
    stats_shots = []
    for file in txt_files:
        try:
            content = download_file_content(drive_service, file['id'])
            if not content.strip():
                logger.warning(f"Empty content in file: {file['name']}")
                continue
            
            logger.debug(f"Content from {file['name']}: {content[:100]}...")  # Log first 100 chars
            stats_shots.append(json.loads(content))
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in file {file['name']}: {str(e)}")
            logger.error(f"Content: {content}")
            continue
        except Exception as e:
            logger.error(f"Error processing file {file['name']}: {str(e)}")
            continue
            
    # Export stats shots to CSV
    stats_df = pd.DataFrame({'stats_shot': stats_shots})
    output_file = f'stats_shots_{subfolder_name}.csv'
    stats_df.to_csv(output_file, index=False)
    logger.info(f"Exported {len(stats_shots)} stats-shot entries to {output_file}")
    
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

def compile_action_labels(drive_service, actions_analysis_folder_id, all_actions, image_index):
    """Compile binary action labels from actions_analysis folders."""
    txt_files = list_txt_files(drive_service, actions_analysis_folder_id)
    action_labels = [0] * len(all_actions)
    for file in txt_files:
        filename = file['name']
        if filename.startswith(f"action_{image_index}_"):
            content = download_file_content(drive_service, file['id'])
            actions_data = json.loads(content).get("actions_by_player", [])
            for action in actions_data:
                if action in all_actions:
                    action_labels[all_actions.index(action)] = 1
    logger.info(f"Compiled action labels for image index {image_index}.")
    return action_labels

def main():
    root_folder_id = '1BdyuWOoHuoeirHS7GwuMe77V3Cd35i_m'
    drive_service = authenticate_gdrive()
    
    logger.info("Starting new data processing pipeline...")
    
    # Compile all unique actions
    all_actions = compile_unique_actions(drive_service, root_folder_id)
    logger.info(f"Total unique actions found: {len(all_actions)}")
    
    # Iterate over each subdirectory in the root folder
    query = f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
    results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    subfolders = results.get('files', [])
    
    # Create separate files for each subfolder
    subfolder_data = []
    
    for subfolder in subfolders:
        logger.info(f"Processing subfolder: {subfolder['name']}")
        
        subfolder_id = subfolder['id']
        query = f"'{subfolder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
        results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        folders = {folder['name']: folder['id'] for folder in results.get('files', [])}
        
        frames_folder_id = folders.get('frames')
        frame_analysis_folder_id = folders.get('frame_analysis')
        actions_analysis_folder_id = folders.get('actions_analysis')
        
        if frames_folder_id and frame_analysis_folder_id and actions_analysis_folder_id:
            image_paths = collect_image_paths(drive_service, frames_folder_id)
            stats_shots = collect_stats_shot(drive_service, frame_analysis_folder_id, subfolder['name'])
            
            # Store data for this subfolder
            subfolder_data.append({
                'name': subfolder['name'],
                'image_paths': image_paths,
                'stats_shots': stats_shots,
                'actions_analysis_folder_id': actions_analysis_folder_id
            })
        else:
            logger.warning(f"Missing required folders in subfolder: {subfolder['name']}")

    # Combine all data into final dataset
    logger.info("Combining all data into final dataset...")
    with open('dataset.csv', mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow(['image_path', 'stats_shot'] + all_actions)
        
        # Write data from each subfolder
        for data in subfolder_data:
            for index, (image_path, stats_shot) in enumerate(zip(data['image_paths'], data['stats_shots']), start=1):
                action_labels = compile_action_labels(
                    drive_service, 
                    data['actions_analysis_folder_id'], 
                    all_actions, 
                    index
                )
                writer.writerow([image_path, json.dumps(stats_shot)] + action_labels)

    logger.info("Dataset compilation complete!")

if __name__ == "__main__":
    main() 