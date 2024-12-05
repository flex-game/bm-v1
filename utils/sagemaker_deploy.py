import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlowModel
import tensorflow as tf
import os
import tarfile
import boto3

def convert_h5_to_saved_model(h5_path, saved_model_dir, preprocessing_info_path):
    # Load the .h5 model
    model = tf.keras.models.load_model(h5_path)
    
    # Save as SavedModel format
    tf.saved_model.save(model, saved_model_dir)
    
    # Create tar.gz file with correct SageMaker structure
    with tarfile.open('model.tar.gz', 'w:gz') as tar:
        # Add all contents from saved_model_dir into model/1/
        for item in os.listdir(saved_model_dir):
            item_path = os.path.join(saved_model_dir, item)
            arcname = os.path.join('model/1', item)
            tar.add(item_path, arcname=arcname)
        # Add preprocessing info at root level
        tar.add(preprocessing_info_path, arcname='preprocessing_info.pkl')
    
    return 'model.tar.gz'

def deploy_model_to_sagemaker():
    # Get the SageMaker execution role
    role = "arn:aws:iam::130323730979:role/SageMakerStandardAllAccessRole"

    # Define the S3 path where your model is stored
    model_data = 's3://bm-v1-bucket/model.tar.gz'

    # Create a TensorFlowModel object
    model = TensorFlowModel(
        model_data=model_data,
        role=role,
        framework_version='2.16.1',  # Updated to match training version
        env={
            'SAGEMAKER_TFS_NGINX_LOGLEVEL': 'info',
            'SAGEMAKER_TFS_WORKER_TIMEOUT_SECS': '300'
        },
        sagemaker_session=sagemaker.Session()
    )

    # Deploy the model to an endpoint
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large'  # Choose an appropriate instance type
    )

    print("Model deployed successfully. Endpoint name:", predictor.endpoint_name)

def inspect_tarfile(tar_path):
    with tarfile.open(tar_path, 'r:gz') as tar:
        print("Contents of", tar_path + ":")
        for member in tar.getmembers():
            print(f"- {member.name}")

if __name__ == "__main__":
    # convert_h5_to_saved_model('trained_model.h5', 'saved_model', 'preprocessing_info.pkl')
    # inspect_tarfile('model.tar.gz')
    deploy_model_to_sagemaker()