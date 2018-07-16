# -*- coding: utf-8 -*-
import torch
import torchvision 
import os

# Hyper-parameters
INPUT_DIM = 784
NUM_CLS = 10
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.0005

# Set up the data directory
data_dir = os.getcwd() + '/data'
DOWNLOAD = True
if os.path.exists(data_dir):
    DOWNLOAD = False
    
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
	root=data_dir,
	train=True,
	transform=torchvision.transforms.ToTensor(),
	download=DOWNLOAD
)
test_dataset = torchvision.datasets.MNIST(
	root=data_dir,
	train=False,
	transform=torchvision.transforms.ToTensor(),
)

# Data loader
train_loader = torch.utils.data.DataLoader(
	dataset=train_dataset,
	batch_size=BATCH_SIZE,
	shuffle=True
)
test_loader = torch.utils.data.DataLoader(
	dataset=test_dataset,
	batch_size=BATCH_SIZE,
	shuffle=False
)

class FC(torch.nn.Module):
	def __init__(self):
		super().__init__()
		block = torch.nn.Sequential(
					torch.nn.Linear(INPUT_DIM, 100),
					torch.nn.ReLU(),
					torch.nn.Linear(100, NUM_CLS),
					torch.nn.Sigmoid()
				)
		self.block = block

	def forward(self, input):
		output = self.block(input)
		return output

# Logistic regression model
model = FC()

# Loss and optimizer
lossFunc = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Train the model
total_step = len(train_loader)
for epoch in range(EPOCHS):
	for idx, (images, labels) in enumerate(train_loader):
		# Reshape the images to (BATCH_SIZE, INPUT_DIM)
		images = images.view(-1, INPUT_DIM)
	
		# Forward pass
		output = model(images)
		loss = lossFunc(output, labels)

		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Print training info
		if (idx+1) % 100 == 0:
			print('Epoch [{}/{}], Step [{}/{}], loss: {:.4f}'
					.format(epoch+1, EPOCHS, idx+1, total_step, loss))

# Test the model (In test phase, we don't need to compute gradients)
with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.view(-1, INPUT_DIM)
		output = model(images)
		_, predicted = torch.max(output.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()

	print('Accuracy of the model on the 10000 test images: {} %'
			.format(100 * correct / total))

# Save the model
torch.save(model.state_dict, 'model.ckpt')
