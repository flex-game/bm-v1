import logging
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, LSTM, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
import numpy as np
from io import BytesIO
from utils.preprocessing import load_action_mapping, preprocess_image, preprocess_text
from utils.s3_utils import s3_get_matching_files, s3_load_data
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import boto3
import pickle

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

# Define the model
def create_model(num_actions, max_sequence_length, num_words, embedding_dim):
    # Image model
    image_input = Input(shape=(224, 224, 3), name='image_input')
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)
    x = GlobalAveragePooling2D()(base_model.output)

    # Text model
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    embedding = Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=max_sequence_length)(text_input)
    y = LSTM(128)(embedding)

    # Combine models
    combined = Concatenate()([x, y])
    z = Dense(256, activation='relu')(combined)
    z = Dense(num_actions, activation='softmax')(z)

    model = Model(inputs=[image_input, text_input], outputs=z)
    return model

def main():
    logger = setup_logging()
    logger.info("=== Starting Model Training Process ===")
    
    try:
        # Load action mapping
        logger.info("Loading action mapping...")
        action_mapping = load_action_mapping()
        num_actions = len(action_mapping)
        logger.info("Found %d unique actions", num_actions)

        # After loading action mapping
        logger.info("Loading and preprocessing training data...")
        logger.info("This may take a while as we process all images and texts...")
        
        # Add progress indicators for data loading
        logger.info("Loading data from S3...")
        raw_images, texts, labels = s3_load_data('bm-v1-training-images', 
                                                'bm-v1-training-text', 
                                                'bm-v1-training-actions', 
                                                common_files)
        logger.info("Loaded %d images, %d texts, and %d labels", 
                   len(raw_images), len(texts), len(labels))
        
        logger.info("3. Creating tokenizer and processing texts...")
        # ... rest of the code ...

        # Hyperparameters
        max_sequence_length = 50
        num_words = 2500
        embedding_dim = 50
        learning_rate = 0.001
        logger.info("Hyperparameters: max_seq_len=%d, vocab_size=%d, embed_dim=%d, lr=%.4f", 
                    max_sequence_length, num_words, embedding_dim, learning_rate)

        # Create model
        logger.info("Creating and compiling model...")
        model = create_model(num_actions, max_sequence_length, num_words, embedding_dim)
        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        
        # Load common files
        logger.info("Finding common files in S3...")
        common_files = s3_get_matching_files('bm-v1-training-images', 
                                           'bm-v1-training-text', 
                                           'bm-v1-training-actions')
        logger.info("Found %d matching files", len(common_files))

        # Preprocess images
        logger.info("Preprocessing images...")
        images = np.array([preprocess_image(BytesIO(img)) for img in raw_images])
        logger.info("Image array shape: %s", images.shape)

        # Tokenizer for text data
        logger.info("Tokenizing text data...")
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(texts)
        logger.info("Vocabulary size: %d", len(tokenizer.word_index))

        # Preprocess texts
        logger.info("Converting texts to sequences...")
        preprocessed_texts = np.array([preprocess_text(txt, tokenizer, max_sequence_length) 
                                     for txt in texts])
        logger.info("Text array shape: %s", preprocessed_texts.shape)

        # Convert labels to categorical (taking only first action)
        logger.info("Processing labels...")
        label_indices = [action_mapping[label[0]] if label else 0 for label in labels]  # Take first action or default to 0
        categorical_labels = tf.keras.utils.to_categorical(label_indices, num_classes=num_actions)
        logger.info("Label array shape: %s", categorical_labels.shape)

        # Add callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath='checkpoints/model_{epoch:02d}.keras',  # NEW - changed .h5 to .keras
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.00001
            )
        ]

        # Train the model
        logger.info("=== Starting Training ===")
        history = model.fit(
            [images, preprocessed_texts],
            categorical_labels,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        # Log final metrics
        logger.info("=== Training Complete ===")
        logger.info("Final training accuracy: %.4f", history.history['accuracy'][-1])
        logger.info("Final validation accuracy: %.4f", history.history['val_accuracy'][-1])

        # Save tokenizer BEFORE model saving
        logger.info("Saving tokenizer...")
        tokenizer_path = '/opt/ml/model/tokenizer.pkl'
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Tokenizer saved to {tokenizer_path}")

        # Save model (only to /opt/ml/model/)
        logger.info("Saving model...")
        model.save('/opt/ml/model/model')
        logger.info("Model saved to /opt/ml/model/model")

        # Add debug logging to verify contents
        logger.info(f"Contents of /opt/ml/model/: {os.listdir('/opt/ml/model/')}")

        logger.info("=== Training Process Complete ===")

    except Exception as e:
        logger.error("Training failed: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()