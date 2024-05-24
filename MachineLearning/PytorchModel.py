import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
from core.MachineLearningModel import MachineLearningModel
import matplotlib.pyplot as plt

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer4(x))
        return x

class PytorchModel(MachineLearningModel):
    def __init__(self, input_size=3, hidden_size=64):  # Adjusted input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BinaryClassifier(input_size, hidden_size).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(categories='auto', sparse_output=False)

    def preprocess_data(self, data):
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)

        # Extract additional time features
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.dayofweek

        # One-hot encode day of the week
        day_of_week_encoded = self.encoder.fit_transform(df[['day']])

        # Combine all features
        feature_columns = ['power_usage', 'hour']
        X = df[feature_columns]
        X = np.hstack((X.values, day_of_week_encoded))
        X = self.scaler.fit_transform(X)
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

    def train(self, data, epochs=10, test_size=0.2, random_state=42):
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X.cpu().numpy(), y.cpu().numpy(), train_size=1-test_size, test_size=test_size, random_state=random_state)
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(self.device), torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(self.device), torch.tensor(y_test, dtype=torch.float32).to(self.device)

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
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_test).float().mean().item()
            print(f'Accuracy: {accuracy * 100:.2f}%')
            print(classification_report(y_test.cpu().numpy(), predictions.cpu().numpy()))

    def predict(self, data):
        X, _ = self.preprocess_data(data)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            predictions = (outputs > 0.5).float().cpu().numpy().flatten().tolist()
        return predictions

    def visualize(self):
        self.model.eval()
        with torch.no_grad():
            feature_importances = self.model.layer1.weight.abs().sum(dim=0).cpu().numpy()
        feature_names = ['power_usage', 'hour'] + [f'day_{i}' for i in range(self.encoder.categories_[0].size)]
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances')
        plt.gca().invert_yaxis()
        plt.show()

    def get_score(self, y, y_pred) -> float:
        # Make the y as long as y_pred
        y = y[:len(y_pred)]
        return accuracy_score(y, y_pred)