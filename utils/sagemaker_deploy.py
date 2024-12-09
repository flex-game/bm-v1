import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlowModel
import tensorflow as tf
import os
import tarfile
import boto3
import datetime

def convert_h5_to_saved_model(h5_path, saved_model_dir, preprocessing_info_path):
    # Use absolute paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h5_path = os.path.join(root_dir, h5_path)
    saved_model_dir = os.path.join(root_dir, saved_model_dir)
    preprocessing_info_path = os.path.join(root_dir, preprocessing_info_path)
    
    print(f"Loading model from: {h5_path}")
    print(f"Saving to: {saved_model_dir}")
    print(f"Preprocessing info at: {preprocessing_info_path}")
    
    print("Loading original model...")
    model = tf.keras.models.load_model(h5_path)
    
    print("\nCreating fresh ResNet50...")
    fresh_resnet = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )
    
    # Freeze the ResNet weights
    fresh_resnet.trainable = False
    
    print("\nRebuilding model with fresh ResNet...")
    # Get the non-ResNet layers from original model
    embedding = model.get_layer('embedding')
    lstm = model.get_layer('lstm')
    dense = model.get_layer('dense')
    
    # Create new inputs
    image_input = tf.keras.Input(shape=(224, 224, 3), name='input_layer')
    text_input = tf.keras.Input(shape=(50,), name='input_layer_2')
    
    # Build new model
    x1 = fresh_resnet(image_input)
    x2 = embedding(text_input)
    x2 = lstm(x2)
    x = tf.keras.layers.Concatenate()([x1, x2])
    outputs = dense(x)
    
    new_model = tf.keras.Model(inputs=[image_input, text_input], outputs=outputs)
    
    # Create concrete function
    print("\nCreating concrete function...")
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name='image_input'),
        tf.TensorSpec(shape=(None, 50), dtype=tf.float32, name='text_input')
    ])
    def serving_fn(image, text):
        # Explicitly name the call
        return new_model([image, text], training=False)
    
    # Force initialization with real-shaped data
    print("\nForcing initialization...")
    dummy_image = tf.zeros((1, 224, 224, 3))
    dummy_text = tf.zeros((1, 50))
    _ = new_model([dummy_image, dummy_text], training=False)
    concrete_func = serving_fn.get_concrete_function()
    
    # Save with explicit naming
    print("\nSaving model...")
    tf.saved_model.save(
        obj=new_model,
        export_dir=saved_model_dir,
        signatures={
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: concrete_func
        },
        options=tf.saved_model.SaveOptions(
            experimental_custom_gradients=False,
            save_debug_info=True
        )
    )
    
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
    
    # Add timestamp to model key
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    key = f'model_{timestamp}.tar.gz'
    
    # Upload the new model to S3
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_tar_path = os.path.join(root_dir, 'model.tar.gz')
    print(f"\nUploading new model to S3...")
    print(f"From: {local_tar_path}")
    print(f"To: s3://{bucket}/{key}")
    s3_client.upload_file(local_tar_path, bucket, key)
    
    # Verify the upload
    response = s3_client.head_object(Bucket=bucket, Key=key)
    print("\nVerifying S3 model:")
    print(f"Last modified: {response['LastModified']}")
    print(f"Size: {response['ContentLength']/1024/1024:.2f} MB")
    
    model_data = f's3://{bucket}/{key}'
    print(f"Using model from: {model_data}")
    
    # Create and deploy model
    model = TensorFlowModel(
        model_data=model_data,
        role=role,
        framework_version='2.16.1',
        env={
            'SAGEMAKER_TFS_NGINX_LOGLEVEL': 'info',
            'SAGEMAKER_TFS_WORKER_TIMEOUT_SECS': '300'
        }
    )
    
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