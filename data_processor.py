import numpy as np
import cv2
from collections import Counter
import matplotlib.pyplot as plt


# class that randomly chooses 1 augmentation method
class DataAugmentor:
    def __init__(self):
        self.augmentation_functions = [self.rotate, self.translate, self.add_noise]

    def rotate(self, image):
        angle = np.random.uniform(-10, 10)
        rows, cols = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        return rotated_image

    def translate(self, image):
        rows, cols = image.shape
        x_translation = np.random.uniform(-5, 5)
        y_translation = np.random.uniform(-5, 5)
        translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
        translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
        return translated_image

    def add_noise(self, image):
        noise_factor = np.random.uniform(0, 0.05)
        noise = np.random.normal(loc=0, scale=noise_factor, size=image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 1)
        return noisy_image

    def augment(self, image):
        selected_function = np.random.choice(self.augmentation_functions)
        augmented_image = selected_function(image)
        return augmented_image


"""
class for data preprocessing, takes X: image dataset, y: corresponding labels, threshold: each class(letter) target 
amount in processed dataset, data_augmentator: class for data augmentation.
"""
class DataProcessor:
    def __init__(self, X, y, num_classes, threshold, data_augmentator):
        self.X = X
        self.y = y
        self.X_processed = []
        self.y_processed = []
        self.threshold = threshold
        self.data_augmentator = data_augmentator
        self.under_sampling_dict = {(tuple([1 if i == j else 0 for i in range(num_classes)])): 0 for j in range(num_classes)}
        self.over_sampling_dict = {(tuple([1 if i == j else 0 for i in range(num_classes)])): threshold for j in range(num_classes)}

    # method for data normalization
    def normalize_data(self):
        self.X = self.X.astype(float) / 255.0

    # method that checks if self.X and self.y contains same amount of values
    def data_info(self):
        if self.y_processed == []:
            print(np.shape(self.X))
            print(np.shape(self.y))
            return np.shape(self.X)[0] == np.shape(self.y)[0]
        else:
            print(np.shape(np.array(self.X_processed)))
            print(np.shape(np.array(self.y_processed)))
            return np.shape(np.array(self.X_processed))[0] == np.shape(np.array(self.y_processed))[0]

    # method for under sampling data to threshold value
    def under_sample(self):
        for idx, label in enumerate(self.y):
            if self.under_sampling_dict[tuple(label)] < self.threshold:
                self.X_processed.append(self.X[self.under_sampling_dict[tuple(label)]])
                self.y_processed.append(label)
                self.under_sampling_dict[tuple(label)] += 1

    # method for over sampling to threshold value, works only after using self.under_sample
    def over_sample(self):
        for key, value in self.over_sampling_dict.items():
            self.over_sampling_dict[key] -= self.under_sampling_dict[key]

        while not all(val == 0 for val in self.over_sampling_dict.values()):
            for idx, label in enumerate(self.y_processed):
                if self.over_sampling_dict[tuple(label)] > 0:
                    letter = self.X_processed[idx]
                    augmented_letter = self.data_augmentator.augment(letter)
                    self.y_processed.append(label)
                    self.X_processed.append(augmented_letter)
                    self.over_sampling_dict[tuple(label)] -= 1
                if all(val == 0 for val in self.over_sampling_dict.values()):
                    break

    def frequency_histogram(self):
        if self.y_processed == []:
            y_tuples = [tuple(arr) for arr in self.y]
        else:
            y_tuples = [tuple(arr) for arr in self.y_processed]

        class_counts = Counter(y_tuples)

        labels = [str(label) for label in class_counts.keys()]
        frequencies = list(class_counts.values())

        plt.figure(figsize=(10, 6))
        plt.bar(labels, frequencies, align='center', alpha=0.7)
        plt.xlabel('Labels')
        plt.ylabel('Frequency')
        plt.title('Label Frequency Histogram')
        plt.xticks(rotation='vertical')
        plt.show()
