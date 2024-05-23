import os

import torch
import torch.nn as nn
import torch.optim as optim

from core.MachineLearningModel import MachineLearningModel


class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.sigmoid(self.layer5(x))
        return x


class PytorchModel(MachineLearningModel):
    def __init__(self, input_size=2, hidden_size=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BinaryClassifier(input_size, hidden_size).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def preprocess_data(self, data):
        # Convert data to PyTorch tensors
        features = torch.tensor([[data['timestamp'][i], data['power_usage'][i]] for i in range(len(data['timestamp']))], dtype=torch.float32)
        labels = torch.tensor(data['appliance_in_use'], dtype=torch.float32)

        # Move data to the device
        features = features.to(self.device)
        labels = labels.to(self.device)

        # Reshape features to have shape (batch_size, input_size)
        features = features.view(-1, self.model.layer1.in_features)

        return features, labels

    def file_extension(self):
        return 'pt'

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, data, epochs=100):
        features, labels = self.preprocess_data(data)
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(features)

            # Remove dimensions of size 1 from outputs
            outputs = outputs.squeeze()

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print loss for this epoch
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, data):
        # Convert data to PyTorch tensors
        features = torch.tensor([[data['timestamp'][i], data['power_usage'][i]] for i in range(len(data['timestamp']))], dtype=torch.float32).to(self.device)
        # Reshape features to have shape (batch_size, input_size)
        features = features.view(-1, self.model.layer1.in_features)

        with torch.no_grad():
            outputs = self.model(features)
            predictions = (outputs > 0.5).float()
        # Convert predictions to a list
        predictions = predictions.cpu().numpy().tolist()
        print(predictions)
        return predictions

    def get_score(self, y, y_pred):
        concat = []
        for i in range(len(y)):
            concat.append([y[i], y_pred[i]])
        return [y == y_pred for y, y_pred in concat].count(True) / len(concat)
