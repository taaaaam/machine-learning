import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from dataset import (
    BATCH_SIZE
)
from model import (
    model,
    device
)

#---EVALUATION
#                 Add the appropriate path for your data (should be a .csv file)
test = pd.read_csv('/home/ubuntu/Documents/digit-recognizer/test.csv',dtype = np.float32)

# split data into features(pixels) and labels(numbers from 0 to 9)
fake_targets = np.zeros(test.shape[0])

features_numpy = test.loc[:,test.columns != "label"].values/255 # normalization

# train test split. Size of train data is 80% and size of test data is 20%. 

features_test = torch.from_numpy(features_numpy)
targets_test = torch.from_numpy(fake_targets).type(torch.LongTensor)

test = torch.utils.data.TensorDataset(features_test,targets_test)

test_loader = DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)

submission = []

#setting model in evaluation mode.
model.eval()

# no gradient is needed
with torch.no_grad():

    for imagesx, _ in test_loader:
                images = imagesx.to(device)
                
                test = Variable(images.view(100,1,28,28))
                
                # Forward propagation
                output = model(test)
                
                _, pred = output.max(dim=1)
                submission.extend([p.item() for p in pred])

sub = pd.DataFrame()
sub['ImageId'] = range(1, len(submission) + 1)
sub['Label'] = submission

sub.to_csv('submission.csv', index = False)
