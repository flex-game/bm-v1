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
    
    # Debug prints
    print("Model summary:")
    model.summary()
    print("\nModel inputs:", model.inputs)
    print("\nModel outputs:", model.outputs)
    
    # After loading model
    resnet_layers = [layer for layer in model.layers if 'resnet' in layer.name.lower()]
    print(f"\nFound {len(resnet_layers)} ResNet layers")
    print("First few ResNet layers:", [l.name for l in resnet_layers[:5]])
    
    # Force eager execution of the model to initialize all variables
    dummy_image = tf.zeros((1, 224, 224, 3))
    dummy_text = tf.zeros((1, 50))
    _ = model([dummy_image, dummy_text], training=False)
    
    # Freeze ResNet50
    model.get_layer('resnet50').trainable = False
    
    # Save with simpler options
    options = tf.saved_model.SaveOptions(experimental_custom_gradients=False)
    tf.saved_model.save(model, saved_model_dir, options=options)
    
    # Create tar.gz file
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
    convert_h5_to_saved_model('trained_model.h5', 'saved_model', 'preprocessing_info.pkl')
    inspect_tarfile('model.tar.gz')
    deploy_model_to_sagemaker()