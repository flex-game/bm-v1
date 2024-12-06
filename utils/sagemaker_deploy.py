import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlowModel
import tensorflow as tf
import os
import tarfile
import boto3

def convert_h5_to_saved_model(h5_path, saved_model_dir, preprocessing_info_path):
    # Use absolute paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h5_path = os.path.join(root_dir, h5_path)
    saved_model_dir = os.path.join(root_dir, saved_model_dir)
    preprocessing_info_path = os.path.join(root_dir, preprocessing_info_path)
    
    print(f"Loading model from: {h5_path}")
    print(f"Saving to: {saved_model_dir}")
    print(f"Preprocessing info at: {preprocessing_info_path}")
    
    # Load the .h5 model
    model = tf.keras.models.load_model(h5_path)
    
    # Debug prints
    print("Model summary:")
    model.summary()
    print("\nModel inputs:", model.inputs)
    print("\nModel outputs:", model.outputs)
    
    # Get the ResNet50 layer and reinitialize it
    print("\nReinitializing ResNet50 weights...")
    resnet = model.get_layer('resnet50')
    
    # Create a fresh ResNet50 with ImageNet weights
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )
    
    # Copy weights layer by layer
    for layer in resnet.layers:
        try:
            matching_layer = base_model.get_layer(layer.name)
            layer.set_weights(matching_layer.get_weights())
            print(f"✓ Copied weights for layer: {layer.name}")
        except Exception as e:
            print(f"✗ Failed to copy weights for layer: {layer.name}")
            print(f"  Error: {str(e)}")
    
    # Save the model
    print("\nSaving model...")
    tf.saved_model.save(model, saved_model_dir)
    
    # Create tar.gz file
    print("\nCreating tar.gz file...")
    with tarfile.open('model.tar.gz', 'w:gz') as tar:
        for item in os.listdir(saved_model_dir):
            item_path = os.path.join(saved_model_dir, item)
            arcname = os.path.join('model/1', item)
            tar.add(item_path, arcname=arcname)
        tar.add(preprocessing_info_path, arcname='preprocessing_info.pkl')
    
    return 'model.tar.gz'

def deploy_model_to_sagemaker():
    # Get the SageMaker execution role
    role = "arn:aws:iam::130323730979:role/SageMakerStandardAllAccessRole"
    
    bucket = 'bm-v1-bucket'
    key = 'model.tar.gz'
    s3_client = boto3.client('s3')
    
    # Verify the model in S3
    response = s3_client.head_object(Bucket=bucket, Key=key)
    print("\nVerifying S3 model:")
    print(f"Last modified: {response['LastModified']}")
    print(f"Size: {response['ContentLength']/1024/1024:.2f} MB")
    
    model_data = f's3://{bucket}/{key}'
    print(f"Using model from: {model_data}")
    
    # Create a TensorFlowModel object
    model = TensorFlowModel(
        model_data=model_data,
        role=role,
        framework_version='2.16.1',
        env={
            'SAGEMAKER_TFS_NGINX_LOGLEVEL': 'info',
            'SAGEMAKER_TFS_WORKER_TIMEOUT_SECS': '300'
        },
        sagemaker_session=sagemaker.Session()
    )
    
    # Deploy the model
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large'
    )
    
    print("\nModel deployed successfully. Endpoint name:", predictor.endpoint_name)

def inspect_tarfile(tar_path):
    with tarfile.open(tar_path, 'r:gz') as tar:
        print("Contents of", tar_path + ":")
        for member in tar.getmembers():
            print(f"- {member.name}")

if __name__ == "__main__":
    convert_h5_to_saved_model('trained_model.h5', 'saved_model', 'preprocessing_info.pkl')
    inspect_tarfile('model.tar.gz')
    deploy_model_to_sagemaker()