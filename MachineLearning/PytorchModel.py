import os
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib

import app
from MachineLearning.classifier.DeepBinaryClassifier import DeepBinaryClassifier
from core.MachineLearningModel import MachineLearningModel
from flask_app import socketio


class PytorchModel(MachineLearningModel):
    def __init__(self, input_size=5, hidden_size=None, learning_rate=0.0005, print_progress=True):
        super().__init__(print_progress)

        # Training parameters
        if hidden_size is None:
            self.hidden_size = [32, 128, 64, 128]
        else:
            self.hidden_size = hidden_size

        self.learning_rate = learning_rate

        self.feature_columns = ['power_usage', 'hour', 'weekday']
        self.function = nn.Sigmoid()
        self.input_size = len(self.feature_columns)
        self.test_size = 0.3
        self.random_state = 23

        # Result parameters
        self.accuracy = 0

        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.metrics = {
            'accuracy': accuracy_score
        }

        self.model = DeepBinaryClassifier(len(self.feature_columns), self.hidden_size, 1).to(self.device)
        self.model.metrics = self.metrics

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scaler = StandardScaler()


    def preprocess_data(self, data, fit_scaler=True):
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Extract additional time features
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday

        # Combine all features
        X = df[self.feature_columns]

        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(df['appliance_in_use'].values, dtype=torch.float32).unsqueeze(1).to(self.device)
        return X, y

    def file_extension(self):
        return 'pt'

    def save_model(self, path):
        save_path = path.replace('.' + self.file_extension(), '')

        model_path = save_path + '/model.' + self.file_extension()
        scaler_path = save_path + '/scaler.pkl'

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

        torch.save(self.model.state_dict(), model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self, path):
        load_path = path.replace('.' + self.file_extension(), '')

        model_path = load_path + '/model.' + self.file_extension()
        scaler_path = load_path + '/scaler.pkl'

        self.model.load_state_dict(torch.load(model_path))
        self.scaler = joblib.load(scaler_path)

    def train(self, data, epochs=100, print_progress=True):
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X.cpu().numpy(), y.cpu().numpy(), train_size=1 - self.test_size,
                                                            test_size=self.test_size, random_state=self.random_state)
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
            if self.print_progress:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
                socketio.emit('training_notification', {'data': f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}'}, namespace='/test')

        self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            predictions = torch.sigmoid(outputs) > 0.05  # Apply sigmoid and thresholding
            accuracy = (predictions == y_test).float().mean().item()
            self.accuracy = accuracy
            print(f'Accuracy: {accuracy * 100:.2f}%')
            print(classification_report(y_test.cpu().numpy(), predictions.cpu().numpy(), zero_division=1))

    def predict(self, data):
        X, _ = self.preprocess_data(data, False)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            predictions = torch.sigmoid(outputs) > 0.05  # Apply sigmoid and thresholding
            return predictions.float().cpu().numpy().flatten().tolist(), (
                        outputs / 5 + 1).float().cpu().numpy().flatten().tolist()

    def get_score(self, y, y_pred) -> float:
        y = y[:len(y_pred)]
        return accuracy_score(y, y_pred)

    def get_document_parameters(self) -> dict:
        return {
            'inputs': ', '.join(self.feature_columns),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'neural_network': self.model.__class__.__name__,
            'function': self.function.__class__.__name__,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'accuracy': self.accuracy
        }


if __name__ == '__main__':
    # Generate a date range
    date_range = pd.date_range(start='2023-01-01', end='2023-01-31', freq='H')

    # Generate random power usage data
    np.random.seed(42)
    power_usage = np.random.normal(loc=100, scale=20, size=len(date_range))

    # Generate random appliance usage data
    appliance_in_use = np.random.choice([0, 1], size=len(date_range), p=[0.7, 0.3])

    # Create a DataFrame
    data = pd.DataFrame({
        'datetime': date_range,
        'power_usage': power_usage,
        'appliance_in_use': appliance_in_use
    })

    model = PytorchModel()
    model.train(data)

    # Prepare an example input for tracing
    example_input, _ = model.preprocess_data(data.head(1))
