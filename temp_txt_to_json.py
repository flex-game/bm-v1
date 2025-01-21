import boto3
import json
import os
import logging
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_s3_client():
    try:
        # Create a session using AWS credentials from environment variables or IAM roles
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        logging.info("S3 client session created successfully.")
        return session.client('s3')
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Error: {e}")
        return None

def txt_to_json(s3, bucket_name, prefix):
    if not s3:
        logging.error("S3 client could not be initialized. Check your AWS credentials.")
        return

    response = list_s3_objects(s3, bucket_name, prefix)
    for obj in response.get('Contents', []):  # Process all objects
        txt_file_path = download_txt_file(s3, bucket_name, obj['Key'])
        json_content = convert_txt_to_json(txt_file_path)
        json_file_path = save_json_to_file(json_content, txt_file_path)
        save_json_locally(json_content, obj['Key'])
        clean_up(txt_file_path)

def list_s3_objects(s3, bucket_name, prefix):
    logging.info(f"Listing objects in bucket '{bucket_name}' with prefix '{prefix}'.")
    return s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

def download_txt_file(s3, bucket_name, key):
    txt_file_path = '/tmp/' + os.path.basename(key)
    s3.download_file(bucket_name, key, txt_file_path)
    logging.info(f"Downloaded file '{key}' to '{txt_file_path}'.")
    return txt_file_path

def convert_txt_to_json(txt_file_path):
    def parse_text_file(content):
        """Parse a single text file's content.
        
        Args:
            content (str): Raw content of text file
        
        Returns:
            list: List of actions by player
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
            logging.warning(f"Error parsing action file: {str(e)}")
            return []

    with open(txt_file_path, 'r') as txt_file:
        content = txt_file.read()
    try:
        actions = parse_text_file(content)
        content_dict = {
            "actions_by_player": actions
        }
        logging.info(f"Successfully converted '{txt_file_path}' to JSON format.")
        return json.dumps(content_dict)
    except json.JSONDecodeError:
        logging.error(f"Failed to convert '{txt_file_path}' to JSON format due to JSONDecodeError.")
        return json.dumps({"content": content})

def save_json_to_file(json_content, txt_file_path):
    json_file_path = txt_file_path.replace('.txt', '.json')
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_content)
    logging.info(f"Converted '{txt_file_path}' to JSON and saved as '{json_file_path}'.")
    return json_file_path

def save_json_locally(json_content, key):
    local_dir = 'local/json_files/'
    os.makedirs(local_dir, exist_ok=True)
    logging.debug(f"Directory created: {local_dir}")
    json_key = os.path.join(local_dir, os.path.basename(key).replace('.txt', '.json'))
    with open(json_key, 'w') as json_file:
        json_file.write(json_content)
    logging.debug(f"File written: {json_key}")

def clean_up(txt_file_path):
    os.remove(txt_file_path)
    logging.info(f"Removed temporary file '{txt_file_path}'.")


def main():
    bucket_name = 'bm-v1-training-actions'
    prefix = ''

    s3 = get_s3_client()

    txt_to_json(s3, bucket_name, prefix)

if __name__ == "__main__":
    main()
