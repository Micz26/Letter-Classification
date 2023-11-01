# imports
import numpy as np

from data_processor import DataAugmentor, DataProcessor
from cnn_model import CnnModel

def main():
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






    CNN = CnnModel(X_train, y_train, X_val, y_val, X_test)
    CNN.create_model()
    CNN.model_summary()
    CNN.train_model()
    CNN.accuracy_plot()
    CNN.check_quantitative_quality()


if 'name' == main():
    main()






"""    Processor = DataProcessor(X_train, y_train, 26, 6000, DataAugmentor())
    Processor.normalize_dataset()
    Processor.frequency_histogram()
    Processor.data_info()
    Processor.under_sample()
    Processor.frequency_histogram()
    Processor.data_info()
    Processor.over_sample()
    Processor.frequency_histogram()
    Processor.data_info()
    X_train, y_train = Processor.__ndarray__()

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
    X_test, y_test = Processor.__ndarray__()"""
