import torch as th
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm
# load data
df = pd.read_csv('mnist/mnist_train.csv')

df_test = pd.read_csv('mnist/mnist_test.csv')

# normalize pixel values and create tensors
labels = th.tensor(df['label'].values)
images = th.tensor(df.drop('label', axis=1).values).float() / 255

labels_test = th.tensor(df['label'].values)
images_test = th.tensor(df.drop('label', axis=1).values).float() / 255

# Define Dataset


class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# create DataLoader
dataset = MNISTDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# create DataLoader for test data
test_dataset = MNISTDataset(images_test, labels_test)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
# Define network


def calculate_accuracy(dataloader):
    correct = 0
    total = 0
    with th.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = th.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define loss function manually


def cross_entropy_loss(y_pred, y_true):
    y_pred = th.log(th.softmax(y_pred, dim=1))
    return -th.mean(th.sum(y_true * y_pred, dim=1))


# Initialize model and optimizer
model = MLP()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# training loop
for epoch in range(10):  # 10 epochs
    for i, (inputs, labels) in enumerate(dataloader):
        # one hot encoding labels for the loss calculation
        labels_one_hot = F.one_hot(labels, num_classes=10).float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = cross_entropy_loss(outputs, labels_one_hot)
        loss.backward()
        optimizer.step()

    accuracy = calculate_accuracy(test_dataloader)
    # print(f"Accuracy on test set: {accuracy*100}%")

    print(
        f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Test Accuracy: {accuracy*100:.2f}%")
print('Finished Training')
