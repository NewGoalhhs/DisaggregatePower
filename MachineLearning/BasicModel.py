import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib
import matplotlib.pyplot as plt
from core.MachineLearningModel import MachineLearningModel


class BasicModel(MachineLearningModel):
    def __init__(self, model=None):
        if model is None:
            self.model = self.get_model()
        else:
            self.model = model

    def preprocess_data(self, data):
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)

        # Generate rolling features for the last 25 power usage values
        for i in range(1, 26):
            df[f'power_usage_lag_{i}'] = df['power_usage'].shift(i)

        # Drop rows with NaN values generated by the rolling window
        df = df.dropna().reset_index(drop=True)

        # Extract additional time features
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.dayofweek

        # Prepare feature set and labels
        feature_columns = [f'power_usage_lag_{i}' for i in range(1, 26)] + ['hour', 'day']
        X = df[feature_columns]
        y = df['appliance_in_use']
        return X, y

    def file_extension(self):
        return 'joblib'

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)

    def train(self, data, test_size=0.2, random_state=42):
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        print("Fitting model...")
        self.model.fit(X_train, y_train)

        print("Predicting...")
        y_pred = self.model.predict(X_test)

        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    def predict(self, power_usages, datetime):
        if len(power_usages) < 25:
            raise ValueError("At least 25 previous power usage values are required for prediction.")

        datetime = pd.to_datetime(datetime)
        hour = datetime.hour
        day = datetime.dayofweek

        # Prepare the feature vector
        features = power_usages[-25:] + [hour, day]
        features = np.array(features).reshape(1, -1)

        prediction = self.model.predict(features)
        return prediction[0]

    @classmethod
    def get_model(cls):
        return RandomForestClassifier(verbose=2, n_jobs=8, n_estimators=1000)

    def visualize(self, feature_columns):
        feature_importances = self.model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances')
        plt.gca().invert_yaxis()
        plt.show()