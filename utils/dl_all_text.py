import boto3
import json
import os

s3 = boto3.client('s3')

def download_all_texts(bucket_name, prefix, output_file):
    text_keys = []
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    for obj in response.get('Contents', []):
        text_keys.append(obj['Key'])

    output_data = []
    for key in text_keys:
        local_path = os.path.join('/tmp', key.split('/')[-1])
        s3.download_file(bucket_name, key, local_path)
        with open(local_path, 'r') as f:
            text_data = json.load(f)
            text_data['filename'] = key.split('/')[-1]  # Add filename key-value pair
            output_data.append(text_data)

    with open(output_file, 'w') as f:
        json.dump(output_data, f)

bucket_name = 'bm-v1-training-text-json'
prefix = ''
output_file = 'data/texts.json'
download_all_texts(bucket_name, prefix, output_file)  