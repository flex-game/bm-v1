import boto3
import os

s3 = boto3.client('s3')

def download_all_images(bucket_name, prefix, output_folder):
    image_keys = []
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    for obj in response.get('Contents', []):
        image_keys.append(obj['Key'])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for key in image_keys:
        local_path = os.path.join(output_folder, key.split('/')[-1])
        s3.download_file(bucket_name, key, local_path)

bucket_name = 'bm-v1-training-images'
prefix = ''
output_folder = 'data/assets'
download_all_images(bucket_name, prefix, output_folder)