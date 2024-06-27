import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib  # For saving and loading the scaler
import json

from MachineLearning.PytorchModel import PytorchModel
from flask_app import socketio

from MachineLearning.classifier.DeepMultiClassifier import DeepMultiClassifier
from core.MachineLearningModel import MachineLearningModel


class AdvancedPytorchModel(MachineLearningModel):
    def __init__(self, input_size=5, hidden_size=None, learning_rate=0.001, output_size=2, print_progress=True):
        super().__init__(print_progress)
        # Training parameters
        if hidden_size is None:
            self.hidden_size = [25, 50, 100, 100]
        else:
            self.hidden_size = hidden_size

        self.learning_rate = learning_rate

        self.feature_columns = ['power_usage', 'hour', 'weekday', 'quarter_hour', 'minute']
        self.function = nn.Sigmoid()
        self.input_size = len(self.feature_columns)
        self.test_size = 0.2
        self.random_state = 42

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

        self.model = DeepMultiClassifier(len(self.feature_columns), self.hidden_size, output_size).to(self.device)
        self.model.metrics = self.metrics

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scaler = StandardScaler()

    def preprocess_data(self, data, fit_scaler=False):
        df = pd.DataFrame(data)
        self.data = df
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')

        # Extract additional time features
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday
        df['quarter_hour'] = df['datetime'].dt.minute // 15
        df['minute'] = df['datetime'].dt.minute

        # Combine all features
        X = df[self.feature_columns]

        # Fit and transform the scaler on the training data
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # Convert to tensor
        X_scaled = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        # Combine all appliance_in_use columns
        appliance_columns = [col for col in df.columns if col.startswith('appliance_in_use_')]
        y = df[appliance_columns].values  # Use the values directly without taking max
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        return X_scaled, y

    def train(self, data, epochs=100, print_progress=True):
        X, y = self.preprocess_data(data, fit_scaler=True)
        X_train, X_test, y_train, y_test = train_test_split(X.cpu().numpy(), y.cpu().numpy(),
                                                            train_size=1 - self.test_size,
                                                            test_size=self.test_size, random_state=self.random_state)
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(self.device), torch.tensor(y_train,
                                                                                                    dtype=torch.float32).to(
            self.device)
        X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(self.device), torch.tensor(y_test,
                                                                                                 dtype=torch.float32).to(
            self.device)

        for epoch in range(epochs):
            socketio.emit('advanced_training_notification', {'title': 'Status update', 'message': f'Epoch {epoch + 1}/{epochs}', 'type': 'info', 'duration': 5000})
            self.model.train()
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.print_progress:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

        socketio.emit('advanced_training_notification', {'title': 'Status update', 'message': 'Finished training', 'type': 'info', 'duration': 5000})

        self.evaluate(X_test, y_test)
        # Save the scaler after training
        joblib.dump(self.scaler, 'scaler.pkl')

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

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            print(outputs)
            predictions = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and thresholding
            print(predictions)
            print(y_test)
            accuracy = (predictions == y_test).float().mean().item()
            self.accuracy = accuracy
            print(f'Accuracy: {accuracy * 100:.2f}%')
            print(classification_report(y_test.cpu().numpy(), predictions.cpu().numpy(), zero_division=1))

    def predict(self, data):
        # Load the scaler if not already loaded
        if not hasattr(self, 'scaler') or not self.scaler:
            self.scaler = joblib.load('scaler.pkl')
        X, _ = self.preprocess_data(data, fit_scaler=False)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            predictions = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and thresholding
            return predictions.float().cpu().numpy().tolist(), outputs.cpu().numpy().tolist()

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
