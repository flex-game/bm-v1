import boto3
import json
import os
import logging
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_s3_client():
    try:
        # Create a session using AWS credentials from environment variables or IAM roles
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        logging.debug("S3 client session created successfully.")
        return session.client('s3')
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Error: {e}")
        return None

def txt_to_json(s3, bucket_name, prefix):
    if not s3:
        logging.error("S3 client could not be initialized. Check your AWS credentials.")
        return

    response = list_s3_objects(s3, bucket_name, prefix)
    logging.debug(f"S3 objects response: {response}")
    for obj in response.get('Contents', []):  
        logging.debug(f"Processing object: {obj['Key']}")
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

def parse_text_file(content):
    """Parse a single text file's content.
    
    Args:
        content (str): Raw content of text file
    
    Returns:
        dict: Parsed JSON-like dictionary
    """
    # Remove markdown code block formatting like we do with actions
    content = content.replace('```json', '').replace('```', '').strip()
    
    # Handle empty files
    if not content:
        return {}
    
    data = {
        "turn": None,
        "science_per_turn": None,
        "culture_per_turn": None,
        "gold_per_turn": None,
        "faith_per_turn": None,
        "military_power": None,
        "city_count": None,
        "unit_count": None,
        "units": [],
        "cities": [],
        "current_research": None,
        "current_civic": None,
        "era_score": None
    }

    # Parse the content to fill the dictionary
    try:
        # Attempt to load the content as JSON
        parsed_content = json.loads(content)
        for key in data.keys():
            if key in parsed_content:
                data[key] = parsed_content[key]
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON content.")
    
    return data

def convert_txt_to_json(txt_file_path):
    logging.debug(f"Converting '{txt_file_path}' to JSON.")
    with open(txt_file_path, 'r') as txt_file:
        content = txt_file.read()

        json_content = parse_text_file(content)
        logging.debug(f"Generated JSON content: {json_content}")
        return json_content

def save_json_to_file(json_content, txt_file_path):
    json_file_path = txt_file_path.replace('.txt', '.json')
    with open(json_file_path, 'w') as json_file:
        json_file.write(json.dumps(json_content))
    logging.info(f"Converted '{txt_file_path}' to JSON and saved as '{json_file_path}'.")
    return json_file_path

def save_json_locally(json_content, key):
    local_dir = 'local/json_files/'
    os.makedirs(local_dir, exist_ok=True)
    logging.debug(f"Directory created: {local_dir}")
    json_key = os.path.join(local_dir, os.path.basename(key).replace('.txt', '.json'))
    with open(json_key, 'w') as json_file:
        json_file.write(json.dumps(json_content))
    logging.debug(f"File written: {json_key}")

def clean_up(txt_file_path):
    os.remove(txt_file_path)
    logging.info(f"Removed temporary file '{txt_file_path}'.")


def main():
    bucket_name = 'bm-v1-training-text'
    prefix = ''

    s3 = get_s3_client()

    txt_to_json(s3, bucket_name, prefix)

if __name__ == "__main__":
    main()
