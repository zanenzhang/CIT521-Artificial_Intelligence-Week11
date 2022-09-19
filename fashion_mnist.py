############################################################
# Imports
############################################################

import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import csv

# Include your imports here, if any are used.


############################################################
# Neural Networks
############################################################

def load_data(file_path, reshape_images):

    if file_path is None:
        return -1

    if reshape_images is None:
        return -1

    labels = []
    pixels = []
    with open(file_path, "r") as file:
        reader = csv.reader(file, delimiter=',')
        next(reader, None)
        for row in reader:
            convertedRow = [int(pixel) for pixel in row]
            labels.append(convertedRow[0])
            pixels.append(convertedRow[1:])
    file.close()

    labels_array = np.array(labels)
    pixels_array = np.array(pixels)

    if reshape_images:
        reshaped_pixels = np.empty((len(pixels_array), 1, 28, 28))
        x = 0
        for image in pixels_array:
            newPixel = np.reshape(image, (1, 28, 28))
            reshaped_pixels[x] = newPixel
            x += 1
        return reshaped_pixels, labels_array

    else:
        return pixels_array, labels_array

# PART 2.2
class EasyModel(torch.nn.Module):
    def __init__(self):
        super(EasyModel, self).__init__()
        self.fc1 = torch.nn.Linear(784, 120)
        self.fc2 = torch.nn.Linear(120, 10)

    def forward(self, x):
        out = F.relu(self.fc1(x))         #The class is callable because the parent class has a special method __call__
        out = self.fc2(out)
        return out

# PART 2.3
class MediumModel(torch.nn.Module):
    def __init__(self):
        super(MediumModel, self).__init__()
        self.fc1 = torch.nn.Linear(784, 480)
        self.fc2 = torch.nn.Linear(480, 240)
        self.drop = torch.nn.Dropout(p=0.2)
        self.fc3 = torch.nn.Linear(240, 120)
        self.fc4 = torch.nn.Linear(120, 10)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.drop(out)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out

# PART 2.4
class AdvancedModel(torch.nn.Module):
    def __init__(self):
        super(AdvancedModel, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=0),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = torch.nn.Linear(in_features=4608, out_features=600)
        self.drop = torch.nn.Dropout(0.25)
        self.fc2 = torch.nn.Linear(in_features=600, out_features=120)
        self.fc3 = torch.nn.Linear(in_features=120, out_features=10)
        #self.fc4 = torch.nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.view(out.size(0), -1)
        #print(out.size())

        out = self.fc1(out)
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        #out = self.fc4(out)

        return out

############################################################
# Fashion MNIST dataset
############################################################

class FashionMNISTDataset(Dataset):
    def __init__(self, file_path, reshape_images):
        self.X, self.Y = load_data(file_path, reshape_images)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

############################################################
# Reference Code
############################################################

def train(model, data_loader, num_epochs, learning_rate):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data_loader):
            images = torch.autograd.Variable(images.float())
            labels = labels.long()       #Critical line here, added manually

            optimizer.zero_grad()
            outputs = model(images.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                y_true, y_predicted = evaluate(model, data_loader)
                print(f'Epoch : {epoch}/{num_epochs}, '
                      f'Iteration : {i}/{len(data_loader)},  '
                      f'Loss: {loss.item():.4f},',
                      f'Train Accuracy: {100.* accuracy_score(y_true, y_predicted):.4f},',
                      f'Train F1 Score: {100.* f1_score(y_true, y_predicted, average="weighted"):.4f}')


def evaluate(model, data_loader):
    model.eval()
    y_true = []
    y_predicted = []
    for images, labels in data_loader:
        images = torch.autograd.Variable(images.float())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels)
        y_predicted.extend(predicted)
    return y_true, y_predicted


def plot_confusion_matrix(cm, class_names, title=None):
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def main():
    class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    num_epochs = 2
    batch_size = 100
    learning_rate = 0.001
    file_path = 'dataset.csv'

    data_loader = torch.utils.data.DataLoader(dataset=FashionMNISTDataset(file_path, False),
                                              batch_size=batch_size,
                                              shuffle=True)
    data_loader_reshaped = torch.utils.data.DataLoader(dataset=FashionMNISTDataset(file_path, True),
                                                       batch_size=batch_size,
                                                       shuffle=True)

    # EASY MODEL
    easy_model = EasyModel()
    train(easy_model, data_loader, num_epochs, learning_rate)
    y_true_easy, y_pred_easy = evaluate(easy_model, data_loader)
    print(f'Easy Model: '
          f'Final Train Accuracy: {100.* accuracy_score(y_true_easy, y_pred_easy):.4f},',
          f'Final Train F1 Score: {100.* f1_score(y_true_easy, y_pred_easy, average="weighted"):.4f}')
    plot_confusion_matrix(confusion_matrix(y_true_easy, y_pred_easy), class_names, 'Easy Model')

    # MEDIUM MODEL
    medium_model = MediumModel()
    train(medium_model, data_loader, num_epochs, learning_rate)
    y_true_medium, y_pred_medium = evaluate(medium_model, data_loader)
    print(f'Medium Model: '
          f'Final Train Accuracy: {100.* accuracy_score(y_true_medium, y_pred_medium):.4f},',
          f'Final F1 Score: {100.* f1_score(y_true_medium, y_pred_medium, average="weighted"):.4f}')
    plot_confusion_matrix(confusion_matrix(y_true_medium, y_pred_medium), class_names, 'Medium Model')

    # ADVANCED MODEL
    advanced_model = AdvancedModel()
    train(advanced_model, data_loader_reshaped, num_epochs, learning_rate)
    y_true_advanced, y_pred_advanced = evaluate(advanced_model, data_loader_reshaped)
    print(f'Advanced Model: '
          f'Final Train Accuracy: {100.* accuracy_score(y_true_advanced, y_pred_advanced):.4f},',
          f'Final F1 Score: {100.* f1_score(y_true_advanced, y_pred_advanced, average="weighted"):.4f}')
    plot_confusion_matrix(confusion_matrix(y_true_advanced, y_pred_advanced), class_names, 'Advanced Model')

if __name__ == '__main__':

    class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    num_epochs = 2

    batch_size = 100

    learning_rate = 0.001

    data_loader_reshaped = torch.utils.data.DataLoader(dataset=FashionMNISTDataset('dataset.csv', True),
                                                       batch_size=batch_size, shuffle=True)

    advanced_model = AdvancedModel()

    train(advanced_model, data_loader_reshaped, num_epochs, learning_rate)

    y_true_advanced, y_pred_advanced = evaluate(advanced_model, data_loader_reshaped)

    print(f'Advanced Model: '
          f'Final Train Accuracy: {100. * accuracy_score(y_true_advanced, y_pred_advanced):.4f},',
          f'Final F1 Score: {100. * f1_score(y_true_advanced, y_pred_advanced, average="weighted"):.4f}')

    """

    class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    num_epochs = 2

    batch_size = 100

    learning_rate = 0.001

    data_loader = torch.utils.data.DataLoader(dataset=FashionMNISTDataset('dataset.csv', False), batch_size=batch_size,
                                              shuffle=True)

    easy_model = EasyModel()

    train(easy_model, data_loader, num_epochs, learning_rate)

    y_true_easy, y_pred_easy = evaluate(easy_model, data_loader)

    print(f'Easy Model: '
          f'Final Train Accuracy: {100. * accuracy_score(y_true_easy, y_pred_easy):.4f},',
          f'Final Train F1 Score: {100. * f1_score(y_true_easy, y_pred_easy, average="weighted"):.4f}')
    
    """