# imports
import tensorflow as tf
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import pickle


# convulutional neural network
# takes tran, evaluate and train datasets
class CnnModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test):
        self.train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(16)
        self.val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(16)
        self.test = X_test
        self.model = None
        self.hist = None
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # model architecture
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

    # model training
    def train_model(self):
        logdir='logs'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        self.hist = self.model.fit(self.train, epochs=5, validation_data=self.val, callbacks=[tensorboard_callback])

    def accuracy_plot(self):
        fig = plt.figure()
        plt.plot(self.hist.history['accuracy'], color='teal', label='accuracy')
        plt.plot(self.hist.history['val_accuracy'], color='orange', label='val_accuracy')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()

    def predict(self, letter):
        return self.model.predict(letter)

    # method for predicting value for random letter from test dataset
    def check_quality(self):
        r_idx = np.random.randint(0, len(self.test))
        letter = self.test[r_idx]
        letter_tensor = tf.convert_to_tensor(letter, dtype=tf.float32)
        plt.imshow(letter_tensor.numpy().astype(int))
        yhat = self.predict(letter.reshape((1, 28, 28, 1)))
        max_val_index = np.argmax(yhat[0])

        print(f'Predicted {self.letters[max_val_index]}')

    # method for checking model performance on test data, results stored in sample_submission.csv
    def check_quantitative_quality(self, file_name):
        original_stdout = sys.stdout
        sys.stdout = open('dummy_file', 'w')

        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['index', 'class'])

            for idx, X in enumerate(self.test):
                X = X.reshape((1, 28, 28, 1))
                yhat = self.predict(X)
                max_val_index = np.argmax(yhat[0])
                writer.writerow([idx, max_val_index])

        sys.stdout = original_stdout

        print(f"Predictions saved in {file_name}")

    def save_model(self):
        with open('letter_classification_model.pkl', 'wb') as model_file:
            pickle.dump(self.model, model_file)



