import pickle
import numpy as np
import pylab
import os
from PIL import Image

path = '/home/dangmc/Documents/DeepLearning/Data/cifar-10/'
image_size = 32
number_channels = 3
number_labels = 10;
train_size = 50000;
test_size = 10000;

def     readFile(filename):
    fo = open(path+filename, 'rb')
    dict = pickle.load(fo)
    fo.close()
    datasets = dict['data'];
    labels = np.array(dict['labels']).reshape(10000);
    return datasets, labels

def	Reshape(datasets, labels):
  datasets = datasets.reshape((-1, image_size, image_size, number_channels)).astype(np.float32)
  labels = (np.arange(number_labels) == labels[:, None]).astype(np.int32)
  return datasets, labels

def make_train_datasets():
    train_dataset = np.ndarray(shape=(train_size, image_size, image_size, number_channels), dtype=np.float32)
    train_labels = np.ndarray(shape=(train_size, number_labels), dtype=np.int32);

    index = 0;
    for id in xrange(5):
        filename = "data_batch_" + str(id + 1);
        data, label = readFile(filename);
        data, label = Reshape(data, label)
        for i in xrange(len(label)):
            train_dataset[index, :, :, :] = data[i, :, :, :];
            train_labels[index, :] = label[i, :]
            index += 1
    return train_dataset, train_labels;


def make_test_datasets():
    test_dataset = np.ndarray(shape=(test_size, image_size, image_size, number_channels), dtype=np.float32)
    test_labels = np.ndarray(shape=(test_size, number_labels), dtype=np.int32);
    filename = "test_batch";
    data, label = readFile(filename);
    data, label = Reshape(data, label)
    for i in xrange(len(label)):
        test_dataset[i, :, :, :] = data[i, :, :, :];
        test_labels[i, :] = label[i, :]
    return test_dataset, test_labels

train_dataset, train_labels = make_train_datasets();
test_dataset, test_labels = make_test_datasets();

print train_dataset.shape
print test_dataset.shape