import sagemaker
import tensorflow as tf
import boto3

s3 = boto3.client('s3')

def download_model(bucket, key, local_path):
    s3.download_file(bucket, key, local_path)

def upload_model(bucket, key, local_path):
    s3.upload_file(local_path, bucket, key)

