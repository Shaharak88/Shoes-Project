# Shoe Classification using Convolutional Neural Networks (CNN)
This project aims to classify women's and men's shoes using Convolutional Neural Networks (CNN). The provided code includes two main classes: CNN and CNNChannel, both implemented in PyTorch. The CNN class handles standard CNN operations, while the CNNChannel class modifies the input to separate and process shoe images in a specific manner.
# Table of Contents
Installation

Usage

Model Architecture

Training

Evaluation

Results




# Installation
Clone the repository:

git clone https://github.com/Shaharak88/Shoes-Project.git
cd Shoes-Project


# Usage
Import the necessary modules and classes:

import torch
from ML_DL_Functions3 import CNN, CNNChannel
# Initialize the model:


    model = CNN()  # For standard CNN
or

    model = CNNChannel()  # For the modified input CNN

# Prepare your data and run the model:


    input_data = torch.randn(1, 3, 448, 224)  # Example input data
    
    output = model(input_data)
    
    print(output)

#Model Architecture
# CNN
The CNN class implements a standard convolutional neural network with the following layers:


Four convolutional layers with ReLU activation and max pooling.

Two fully connected layers to produce the final classification.

# CNNChannel
The CNNChannel class processes the input differently by splitting the input image along the height dimension and concatenating the halves along the channel dimension before passing it through a similar architecture as CNN.

# Training
Prepare your training and validation datasets.

Define your loss function and optimizer:
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model:


    for epoch in range(num_epochs):

        for inputs, labels in train_loader:
        
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            optimizer.step()
        
# Evaluation

Evaluate the model on the test dataset:

    model.eval()
    
    with torch.no_grad():

    for inputs, labels in test_loader:
    
        outputs = model(inputs)
        
        _, predicted = torch.max(outputs, 1)
        
        # Calculate accuracy and other metrics
# validation:

![image](https://github.com/Shaharak88/Shoes-Project/assets/95345116/3050582f-d3f5-4dfc-8452-3bf9f29414c3)


# Results
Men's Shoes example:

Accuracy: 81.7

Women's Shoes example:

Accuracy: 83.3

# Example for Men's Shoes the model classifid correctly (The shoes are the same pair, but the model classified them as not the same pair.):
    display(1,test_m_data)
    
![image](https://github.com/Shaharak88/Shoes-Project/assets/95345116/f9713230-7be9-46dd-819d-f5f59fa99c7e)

# Example for Men's Shoes the model classifid incorrectly (The shoes are the same pair, but the model classified them as not the same pair.):
![image](https://github.com/Shaharak88/Shoes-Project/assets/95345116/6f1ce4dc-74af-47af-9cc7-ec19b1debb0e)
