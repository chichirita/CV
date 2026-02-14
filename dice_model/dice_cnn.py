import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

class DiceClassifier:
    def __init__(self, image_size=128):
        self.image_size = image_size
        self.model = None
        self.history = None

    def build_model(self):
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu',
                         input_shape=(self.image_size, self.image_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),

            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),

            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(6, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def load_and_preprocess_data(self, dataset_path='dice_dataset'):
        """Загружает и подготавливает данные"""
        images = []
        labels = []

        for number in range(1, 7):
            class_dir = os.path.join(dataset_path, str(number))
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)

                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.image_size, self.image_size))
                img = img.astype('float32') / 255.0

                images.append(img)
                labels.append(number - 1)

        X = np.array(images)
        y = keras.utils.to_categorical(labels, 6)

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=labels)

    def train(self, X_train, X_val, y_train, y_val, epochs=1):
        """Обучает модель"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7
            )
        ]

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.history.history['accuracy'], label='Точность на обучении')
        ax1.plot(self.history.history['val_accuracy'], label='Точность на валидации')
        ax1.set_title('Точность модели')
        ax1.set_xlabel('Эпоха')
        ax1.set_ylabel('Точность')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.history.history['loss'], label='Потери на обучении')
        ax2.plot(self.history.history['val_loss'], label='Потери на валидации')
        ax2.set_title('Потери модели')
        ax2.set_xlabel('Эпоха')
        ax2.set_ylabel('Потери')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


classifier = DiceClassifier()
model = classifier.build_model()

model.summary()
