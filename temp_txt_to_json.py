import boto3
import json
import os

s3 = boto3.client('s3')

def txt_to_json(bucket_name, prefix):
    # List all files in the specified S3 bucket and prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    for obj in response.get('Contents', []):
        # Download the .txt file
        txt_file_path = '/tmp/' + os.path.basename(obj['Key'])
        s3.download_file(bucket_name, obj['Key'], txt_file_path)
        
        # Read the content of the .txt file
        with open(txt_file_path, 'r') as txt_file:
            content = txt_file.read()
        
        # Convert the content to JSON format
        json_content = json.dumps({"content": content})
        json_file_path = txt_file_path.replace('.txt', '.json')
        
        # Save the JSON content to a new file
        with open(json_file_path, 'w') as json_file:
            json_file.write(json_content)
        # Clean the content before saving as JSON
        try:
            content_dict = json.loads(content)
            if "actions_by_player" in content_dict:
                json_content = json.dumps(content_dict["actions_by_player"])
            else:
                json_content = json.dumps(content_dict)
        except json.JSONDecodeError:
            json_content = json.dumps({"error": "Invalid JSON content"})
        
        # Save the .json file to a tmp local directory instead of re-uploading
        tmp_local_dir = '/tmp/json_files/'
        os.makedirs(tmp_local_dir, exist_ok=True)
        json_key = os.path.join(tmp_local_dir, os.path.basename(obj['Key']).replace('.txt', '.json'))
        
        with open(json_key, 'w') as json_file:
            json_file.write(json_content)
        # Clean up temporary files
        os.remove(txt_file_path)
        os.remove(json_file_path)

bucket_name = 'bm-v1-training-actions'
prefix = ''

txt_to_json(bucket_name, prefix)
