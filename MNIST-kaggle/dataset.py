import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 100
N_ITERS = 10000

#---DATASET
# Prepare Dataset
# load data         Add the appropriate path for your data (should be a .csv file)
train = pd.read_csv('/home/ubuntu/Documents/digit-recognizer/train.csv',dtype = np.float32)

# split data into features(pixels) and labels(numbers from 0 to 9)
targets_numpy = train.label.values
features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization


print(features_numpy.shape)

# train test split. Size of train data is 80% and size of test data is 20%. 
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                             targets_numpy,
                                                                             test_size = 0.2,
                                                                             random_state = 42) 

features_train = torch.from_numpy(features_train)
targets_train = torch.from_numpy(targets_train).type(torch.LongTensor)
features_test = torch.from_numpy(features_test)
targets_test = torch.from_numpy(targets_test).type(torch.LongTensor)

#print(features_train)
#print(targets_train)
# batch_size, epoch and iteration

NUM_EPOCHS = N_ITERS / (len(features_train) / BATCH_SIZE)
NUM_EPOCHS = int(NUM_EPOCHS)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(features_train,targets_train)
test = torch.utils.data.TensorDataset(features_test,targets_test)

# data loader
train_loader = DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
test_loader = DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)

# visualize one of the images in data set
plt.imshow(features_numpy[10].reshape(28,28))
plt.axis("off")
plt.title(str(targets_numpy[10]))
plt.savefig('graph.png')
plt.show()
