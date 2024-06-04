import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

import app
from MachineLearning.classifier.DeepBinaryClassifier import DeepBinaryClassifier
from core.MachineLearningModel import MachineLearningModel

class PytorchModel(MachineLearningModel):
    def __init__(self, input_size=5, hidden_size=None):
        if hidden_size is None:
            hidden_size = [64, 32, 16]

        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = DeepBinaryClassifier(input_size, hidden_size, 1).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scaler = StandardScaler()

    def preprocess_data(self, data):
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Extract additional time features
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday

        # Combine all features
        feature_columns = ['power_usage', 'month', 'day', 'hour','weekday']
        X = df[feature_columns]
        X = self.scaler.fit_transform(X)
        # TODO: Kijken waarom die hier op zijn bek gaat
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(df['appliance_in_use'].values, dtype=torch.float32).unsqueeze(1).to(self.device)
        return X, y

    def file_extension(self):
        return 'pt'

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, data, epochs=100, test_size=0.2, random_state=42):
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X.cpu().numpy(), y.cpu().numpy(), train_size=1 - test_size,
                                                            test_size=test_size, random_state=random_state)
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(self.device), torch.tensor(y_train,
                                                                                                    dtype=torch.float32).to(
            self.device)
        X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(self.device), torch.tensor(y_test,
                                                                                                 dtype=torch.float32).to(
            self.device)

        for epoch in range(epochs):
            self.model.train()
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

        self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            predictions = torch.sigmoid(outputs) > 0.05  # Apply sigmoid and thresholding
            accuracy = (predictions == y_test).float().mean().item()
            print(f'Accuracy: {accuracy * 100:.2f}%')
            print(classification_report(y_test.cpu().numpy(), predictions.cpu().numpy(), zero_division=1))

    def predict(self, data):
        X, _ = self.preprocess_data(data)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            predictions = torch.sigmoid(outputs) > 0.05  # Apply sigmoid and thresholding
            return predictions.float().cpu().numpy().flatten().tolist(), (outputs / 5 + 1).float().cpu().numpy().flatten().tolist()

    def get_score(self, y, y_pred) -> float:
        y = y[:len(y_pred)]
        return accuracy_score(y, y_pred)


if __name__ == '__main__':
    df = pd.read_csv(app.__ROOT__ + '/QuickRunData/synthetic_power_usage_data.csv')
    data = {
        'datetime': df['datetime'],
        'power_usage': df['power_usage'],
        'appliance_in_use': df['appliance_in_use']
    }
    model = PytorchModel()
    model.train(data)
    predictions = model.predict(data)
    print(data['datetime'][:10])
    print(data['power_usage'][:10])
    print(data['appliance_in_use'][:10])
    print(predictions[:10])
    score = model.get_score(data['appliance_in_use'], predictions)
    print(score)
