import boto3
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_action_file(content):
    """Parse a single action file's content.
    
    Args:
        content (str): Raw content of action file
    
    Returns:
        list: List of actions from the file
    """
    try:
        # Remove markdown code block formatting
        content = content.replace('```json', '').replace('```', '').strip()
        
        # Handle empty files
        if not content:
            return []
            
        action_data = json.loads(content)
        
        # Handle case where action_data might be None
        if action_data is None:
            return []
            
        # Handle case where actions_by_player might be None
        return action_data.get('actions_by_player', [])
        
    except Exception as e:
        logger.warning(f"Error parsing action file: {str(e)}")
        return []

def prepare_action_labels(force_refresh=False):
    """Extract and prepare unique action labels from training files.
    
    Args:
        force_refresh (bool): If True, regenerate mapping even if it exists.
    
    Returns:
        tuple: (action_mapping dict, number of unique actions)
    """
    s3_client = boto3.client('s3')
    model_bucket = 'bm-v1-model'
    
    # Check for existing mapping unless force refresh
    if not force_refresh:
        try:
            response = s3_client.get_object(Bucket=model_bucket, Key='action_mapping.json')
            action_mapping = json.loads(response['Body'].read().decode('utf-8'))
            logger.info("Using existing action mapping")
            return action_mapping, len(action_mapping)
        except s3_client.exceptions.NoSuchKey:
            logger.info("No existing action mapping found, generating new one")
    
    actions_bucket = 'bm-v1-training-actions'
    unique_actions = set()
    
    # List all files in actions bucket
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=actions_bucket):
        for obj in page['Contents']:
            response = s3_client.get_object(Bucket=actions_bucket, Key=obj['Key'])
            content = response['Body'].read().decode('utf-8')
            
            # Use the separated parsing function
            actions = parse_action_file(content)
            for action in actions:
                unique_actions.add(action)
    
    # Convert to sorted list for consistent ordering
    action_list = sorted(list(unique_actions))
    action_mapping = {action: idx for idx, action in enumerate(action_list)}
    
    # Save action mapping to S3
    s3_client.put_object(
        Bucket='bm-v1-model',
        Key='action_mapping.json',
        Body=json.dumps(action_mapping)
    )
    
    logger.info(f"Prepared action labels: {action_mapping}")
    return action_mapping, len(action_list) 

def load_action_mapping():
    """Load action mapping from S3.
    
    Returns:
        dict: Mapping of action names to indices
    """
    s3_client = boto3.client('s3')
    try:
        response = s3_client.get_object(Bucket='bm-v1-model', Key='action_mapping.json')
        action_mapping = json.loads(response['Body'].read().decode('utf-8'))
        logger.info("Loaded action mapping successfully")
        return action_mapping
    except Exception as e:
        logger.error(f"Error loading action mapping: {str(e)}")
        raise 