from dataset import (
    train_loader,
    test_loader,
    NUM_EPOCHS,
)
from model import (
    device,
    optimizer,
    error,
    model
)
import torch
from torch.autograd import Variable

#---TRAINING
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(NUM_EPOCHS):
    for i, (imagesx, labelsx) in enumerate(train_loader):
        images = imagesx.to(device)
        labels = labelsx.to(device)
        train = Variable(images.view(100,1,28,28))
        labels = Variable(labels)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        if count % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for imagesx, labelsx in test_loader:
                images = imagesx.to(device)
                labels = labelsx.to(device)
                test = Variable(images.view(100,1,28,28))
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))