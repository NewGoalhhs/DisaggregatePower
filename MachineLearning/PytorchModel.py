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
        x = self.sigmoid(self.layer2(x))
        return x


class PytorchModel(MachineLearningModel):
    def __init__(self, input_size=1, hidden_size=10):
        self.model = BinaryClassifier(input_size, hidden_size)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess_data(self, data):
        # Implement your data preprocessing here
        pass

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, data, epochs=100):
        print('Training model...')
        print(data)
        labels = list(data.keys())
        data = list(data.values())

        print('Labels: ')
        print(labels)
        print('Data: ')
        print(data)

        # Convert data to PyTorch tensors
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        # Move data to the device
        data = data.to(self.device)
        labels = labels.to(self.device)

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(data)

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print loss for this epoch
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, data):
        with torch.no_grad():
            outputs = self.model(data)
            predictions = (outputs > 0.5).float()
        return predictions
