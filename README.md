# Shoe Classification using Convolutional Neural Networks (CNN)
This project aims to classify women's and men's shoes using Convolutional Neural Networks (CNN). The provided code includes two main classes: CNN and CNNChannel, both implemented in PyTorch. The CNN class handles standard CNN operations, while the CNNChannel class modifies the input to separate and process shoe images in a specific manner.
# Table of Contents
Installation
Usage
Model Architecture
Training
Evaluation
Results
Contributing
License
Installation
Clone the repository:

sh
Copy code
git clone https://github.com/Shaharak88/Shoes-Project.git
cd Shoes-Project


Usage
Import the necessary modules and classes:

import torch
from ML_DL_Functions3 import CNN, CNNChannel
Initialize the model:

python
Copy code
model = CNN()  # For standard CNN
# or
model = CNNChannel()  # For the modified input CNN
Prepare your data and run the model:

python
Copy code
input_data = torch.randn(1, 3, 448, 224)  # Example input data
output = model(input_data)
print(output)
Model Architecture
CNN
The CNN class implements a standard convolutional neural network with the following layers:

Four convolutional layers with ReLU activation and max pooling.
Two fully connected layers to produce the final classification.
CNNChannel
The CNNChannel class processes the input differently by splitting the input image along the height dimension and concatenating the halves along the channel dimension before passing it through a similar architecture as CNN.

Training
Prepare your training and validation datasets.
Define your loss function and optimizer:
python
Copy code
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
Train the model:
python
Copy code
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
Evaluation
Evaluate the model on the test dataset:
python
Copy code
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        # Calculate accuracy and other metrics
Results
Example results for shoe classification:

Sample Image	Predicted Class	Actual Class
Men's Shoe	Men's Shoe
Women's Shoe	Women's Shoe
Accuracy Graph

Loss Graph

Contributing
Contributions are welcome! Please fork this repository and submit pull requests.
