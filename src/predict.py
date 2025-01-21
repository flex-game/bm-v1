import logging
import sagemaker
import tensorflow as tf
import boto3

s3 = boto3.client('s3')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def download_model(bucket, key, local_path):
    s3.download_file(bucket, key, local_path)

def upload_model(bucket, key, local_path):
    s3.upload_file(local_path, bucket, key)

def host_model(model, vectorizer):
    return model, vectorizer

def predict_move(image, text, model, vectorizer):
    return str(move)

def train_model() -> Any:
    logger = setup_logging()
    logger.info("=== Starting Train ===")

    # Load and preprocess data
    bucket = 'bm-v1'

    # Get action keys
    action_keys = []
    response = s3.list_objects_v2(Bucket='bm-v1-training-actions-json')
    for obj in response.get('Contents', []):
        action_keys.append(obj['Key'])

    # Get image keys
    image_keys = []
    response = s3.list_objects_v2(Bucket='bm-v1-training-images')
    for obj in response.get('Contents', []):
        image_keys.append(obj['Key'])

    # Get text keys
    text_keys = []
    response = s3.list_objects_v2(Bucket='bm-v1-training-texts-json')
    for obj in response.get('Contents', []):
        text_keys.append(obj['Key'])

    local_dir = 'data/training'

    # Define the model
    image_input = tf.keras.layers.Input(shape=(224, 224, 3), name='image_input')
    text_input = tf.keras.layers.Input(shape=(300,), name='text_input')

    resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=image_input)
    resnet_output = tf.keras.layers.GlobalAveragePooling2D()(resnet.output)

    concatenated = tf.keras.layers.concatenate([resnet_output, text_input])
    dense1 = tf.keras.layers.Dense(512, activation='relu')(concatenated)
    dense2 = tf.keras.layers.Dense(256, activation='relu')(dense1)
    output = tf.keras.layers.Dense(300, activation='softmax')(dense2)

    model = tf.keras.models.Model(inputs=[image_input, text_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit([images, texts], actions, epochs=10, batch_size=32, validation_split=0.2)

    # Save the model
    model.save('/tmp/model.h5')
    upload_model(bucket, 'path/to/save/model.h5', '/tmp/model.h5')

    logger.info("=== Training Process Complete ===")

    return model

def main():

    return
    
if __name__ == "__main__":
    main()