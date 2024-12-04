import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlowModel

def deploy_model_to_sagemaker():
    # Get the SageMaker execution role
    role = get_execution_role()

    # Define the S3 path where your model is stored
    model_data = 's3://your-unique-bucket-name/model/model.tar.gz'

    # Create a TensorFlowModel object
    model = TensorFlowModel(
        model_data=model_data,
        role=role,
        framework_version='2.4.0',  # Ensure this matches your model's TensorFlow version
        sagemaker_session=sagemaker.Session()
    )

    # Deploy the model to an endpoint
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large'  # Choose an appropriate instance type
    )

    print("Model deployed successfully. Endpoint name:", predictor.endpoint_name)

if __name__ == "__main__":
    deploy_model_to_sagemaker() 