import boto3
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_action_labels():
    """Extract and prepare unique action labels from training files."""
    s3_client = boto3.client('s3')
    actions_bucket = 'bm-v1-training-actions'
    unique_actions = set()
    
    # List all files in actions bucket
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=actions_bucket):
        for obj in page['Contents']:
            response = s3_client.get_object(Bucket=actions_bucket, Key=obj['Key'])
            content = response['Body'].read().decode('utf-8')
            
            # Remove markdown code block formatting
            content = content.replace('```json', '').replace('```', '').strip()
            
            action_data = json.loads(content)
            
            # Handle case where actions_by_player might be None
            actions = action_data.get('actions_by_player', [])
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