import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from ClonedAudioDetector import CNNClassifier


def process(model: CNNClassifier,
            training_data: DataLoader,
            validation_data: DataLoader):
    """trains a cnn classifier on given data and returns results"""
    print("test")
    return train_model(model, training_data, validation_data, use_coda_if_available=True)


def train_model(model: CNNClassifier,
                training_data_loader: DataLoader,
                validation_data_loader: DataLoader,
                use_coda_if_available: bool):
    """trains a cnn classifier on given data and returns results"""
    device = "cpu"
    if use_coda_if_available and torch.cuda.is_available():
        device = "cuda"

    epochs = 10

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters())

    print(validation_data_loader)

    for epoch in range(epochs):
        model.train()

        print(epoch)

        running_loss = 0.0
        for images, labels in training_data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            print(images.shape)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
