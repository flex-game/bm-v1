import os
import csv
import json
import logging
from gdrive_utils import authenticate_gdrive, list_txt_files, list_jpg_files, download_file_content

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

def collect_stats_shot(drive_service, frame_analysis_folder_id):
    """Collect stats-shot data from frame_analysis folders."""
    txt_files = list_txt_files(drive_service, frame_analysis_folder_id)
    stats_shots = []
    for file in txt_files:
        content = download_file_content(drive_service, file['id'])
        stats_shots.append(json.loads(content))
    logger.info(f"Collected {len(stats_shots)} stats-shot entries from frame_analysis folder.")
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
                content = download_file_content(drive_service, file['id'])
                actions_data = json.loads(content).get("actions_by_player", [])
                all_actions.update(actions_data)
    
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
            stats_shots = collect_stats_shot(drive_service, frame_analysis_folder_id)
            
            csv_file_path = 'dataset.csv'
            with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                for index, (image_path, stats_shot) in enumerate(zip(image_paths, stats_shots), start=1):
                    action_labels = compile_action_labels(drive_service, actions_analysis_folder_id, all_actions, index)
                    writer.writerow([image_path, json.dumps(stats_shot)] + action_labels)
        else:
            logger.warning(f"Missing required folders in subfolder: {subfolder['name']}")

if __name__ == "__main__":
    main() 