"""
Training script for CNN classification model
Trains model to classify cardio health check kit results (Positive, Negative, Invalid)
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultClassificationTrainer:
    """Trainer for result classification using CNN"""
    
    def __init__(self, data_path: str = "data", img_size: tuple = (224, 224)):
        """
        Initialize trainer
        
        Args:
            data_path: Path to training data
            img_size: Input image size for the model
        """
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.model = None
        self.class_names = ['Positive', 'Negative', 'Invalid']
        self.num_classes = len(self.class_names)
        
    def load_data(self):
        """Load and preprocess training data"""
        try:
            images = []
            labels = []
            
            # Load images from each class directory
            for class_idx, class_name in enumerate(self.class_names):
                class_path = self.data_path / "classification" / class_name.lower()
                
                if not class_path.exists():
                    logger.warning(f"Class directory not found: {class_path}")
                    continue
                
                # Load images from class directory
                for img_file in class_path.glob("*.jpg"):
                    try:
                        # Load and preprocess image
                        img = cv2.imread(str(img_file))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.img_size)
                        img = img.astype(np.float32) / 255.0
                        
                        images.append(img)
                        labels.append(class_idx)
                        
                    except Exception as e:
                        logger.warning(f"Error loading image {img_file}: {e}")
                        continue
                
                logger.info(f"Loaded {len(list(class_path.glob('*.jpg')))} images for class {class_name}")
            
            if not images:
                raise ValueError("No training images found")
            
            # Convert to numpy arrays
            X = np.array(images)
            y = np.array(labels)
            
            # Convert labels to categorical
            y_categorical = keras.utils.to_categorical(y, self.num_classes)
            
            logger.info(f"Loaded {len(X)} images with {self.num_classes} classes")
            return X, y_categorical, y
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_model(self):
        """Create CNN model architecture"""
        try:
            # Use MobileNetV2 as base architecture for mobile optimization
            base_model = keras.applications.MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Freeze base model layers
            base_model.trainable = False
            
            # Add custom classification head
            model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.2),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            logger.info("Model architecture created")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
    
    def train_model(self, X, y, epochs: int = 50, batch_size: int = 32):
        """
        Train the classification model
        
        Args:
            X: Training images
            y: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        try:
            # Split data into train/validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
            )
            
            # Data augmentation
            train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode='nearest'
            )
            
            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                ),
                callbacks.ModelCheckpoint(
                    'models/classification/best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False
                )
            ]
            
            # Train model
            history = self.model.fit(
                train_datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks_list,
                verbose=1
            )
            
            logger.info("Model training completed")
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test images
            y_test: Test labels
        """
        try:
            # Make predictions
            predictions = self.model.predict(X_test)
            y_pred = np.argmax(predictions, axis=1)
            y_true = np.argmax(y_test, axis=1)
            
            # Calculate metrics
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            
            # Classification report
            report = classification_report(
                y_true, y_pred, 
                target_names=self.class_names,
                output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            logger.info(f"Test Accuracy: {test_accuracy:.4f}")
            logger.info(f"Test Loss: {test_loss:.4f}")
            
            return {
                'accuracy': test_accuracy,
                'loss': test_loss,
                'classification_report': report,
                'confusion_matrix': cm,
                'predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def export_model(self, model_path: str, format: str = "tflite"):
        """
        Export model to different formats
        
        Args:
            model_path: Path to trained model
            format: Export format (tflite, onnx, etc.)
        """
        try:
            if format == "tflite":
                # Convert to TensorFlow Lite
                converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                
                tflite_model = converter.convert()
                
                # Save TFLite model
                tflite_path = model_path.replace('.h5', '.tflite')
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                
                logger.info(f"TFLite model saved to {tflite_path}")
                return tflite_path
                
            elif format == "onnx":
                # Convert to ONNX
                import tf2onnx
                spec = (tf.TensorSpec((None, *self.img_size, 3), tf.float32, name="input"),)
                onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature=spec, opset=13)
                
                onnx_path = model_path.replace('.h5', '.onnx')
                with open(onnx_path, 'wb') as f:
                    f.write(onnx_model.SerializeToString())
                
                logger.info(f"ONNX model saved to {onnx_path}")
                return onnx_path
                
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            raise
    
    def plot_training_history(self, history):
        """Plot training history"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot accuracy
            ax1.plot(history.history['accuracy'], label='Training Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            
            # Plot loss
            ax2.plot(history.history['loss'], label='Training Loss')
            ax2.plot(history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig('models/classification/training_history.png')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")

def main():
    """Main training pipeline"""
    trainer = ResultClassificationTrainer()
    
    try:
        # Create models directory
        os.makedirs('models/classification', exist_ok=True)
        
        # Load data
        X, y_categorical, y = trainer.load_data()
        logger.info("Data loading completed")
        
        # Create model
        model = trainer.create_model()
        logger.info("Model creation completed")
        
        # Train model
        history = trainer.train_model(X, y_categorical, epochs=30, batch_size=16)
        logger.info("Model training completed")
        
        # Evaluate model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y
        )
        
        evaluation_results = trainer.evaluate_model(X_test, y_test)
        logger.info("Model evaluation completed")
        
        # Export models
        model_path = 'models/classification/best_model.h5'
        if os.path.exists(model_path):
            tflite_path = trainer.export_model(model_path, "tflite")
            logger.info(f"TFLite model exported to {tflite_path}")
        
        # Plot training history
        trainer.plot_training_history(history)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
