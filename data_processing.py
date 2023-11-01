# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from data_processor import DataAugmentor, DataProcessor

X_train = np.load('C:\\Users\\mikol\\Downloads\\X_train.npy.zip')
y_train = np.load('C:\\Users\\mikol\\Downloads\\y_train.npy.zip')
X_val = np.load('C:\\Users\\mikol\\Downloads\\X_val.npy.zip')
y_val = np.load('C:\\Users\\mikol\\Downloads\\y_val.npy')
X_test = np.load('C:\\Users\\mikol\\Downloads\\X_test.npy.zip')

y_train = y_train['y_train']
X_train = X_train['X_train']

y_val = y_val
X_val = X_val['X_val']

X_test = X_test['X_test']


Processor = DataProcessor(X_train, y_train, 26, 6000, DataAugmentor())
Processor.normalize_dataset()
Processor.frequency_histogram()
Processor.data_info()
Processor.under_sample()
Processor.frequency_histogram()
Processor.data_info()
Processor.over_sample()
Processor.frequency_histogram()
Processor.data_info()

Processor = DataProcessor(X_val, y_val, 26, 1700, DataAugmentor())
Processor.normalize_dataset()
Processor.frequency_histogram()
Processor.data_info()
Processor.under_sample()
Processor.frequency_histogram()
Processor.data_info()
Processor.over_sample()
Processor.frequency_histogram()
Processor.data_info()
