# imports
import tensorflow as tf
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle


class CnnModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test):
        self.train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(16)
        self.val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(16)
        self.test = X_test
        self.model = None
        self.hist = None
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def normalize_letters(self, letters):
        return letters.astype(np.float32) / 255.0

    def create_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=l2(0.01)))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(28, activation='relu', kernel_regularizer=l2(0.01)))
        self.model.add(tf.keras.layers.Dense(26, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def model_summary(self):
        self.model.summary()

    def train_model(self):
        logdir='logs'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        self.hist = self.model.fit(self.train, epochs=1, validation_data=self.val, callbacks=[tensorboard_callback])

    def accuracy_plot(self):
        fig = plt.figure()
        plt.plot(self.hist.history['accuracy'], color='teal', label='accuracy')
        plt.plot(self.hist.history['val_accuracy'], color='orange', label='val_accuracy')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()

    def predict(self, letter):
        return self.model.predict(letter)

    def check_quality(self):
        letter = np.random.choice(self.test)
        resize = tf.image.resize(letter, (28, 28))
        plt.imshow(resize.numpy().astype(int))
        yhat = self.predict(np.expand_dims(resize / 255, 0))
        max_val_index = np.argmax(yhat[0])

        print(f'Letter is {self.letters[max_val_index]}')

    def check_quantitative_quality(self):
        self.test = self.normalize_letters(self.test)
        with open('sample_submission.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['index', 'prediction'])

            for idx, X in enumerate(self.test):
                yhat = self.predict(X)
                writer.writerow([idx, yhat])

        print("Predictions saved in sample_submission.csv")

    def save_model(self):
        with open('letter_classification_model.pkl', 'wb') as model_file:
            pickle.dump(self.model, model_file)


