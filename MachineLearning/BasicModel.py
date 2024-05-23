import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib
import matplotlib.pyplot as plt

import app
from core.Database import Database
from SQL.SQLQueries import DatabaseOperations as Query
from core.MachineLearningModel import MachineLearningModel


class BasicModel(MachineLearningModel):
    def __init__(self, model=None):
        if model is None:
            self.model = self.get_model()
        else:
            self.model = model

    def preprocess_data(self, data):
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp').reset_index(drop=True)

        # Generate rolling features for the last 25 power usage values
        for i in range(1, 26):
            df[f'power_usage_lag_{i}'] = df['power_usage'].shift(i)

        # Drop rows with NaN values generated by the rolling window
        df = df.dropna().reset_index(drop=True)

        # Extract additional time features
        # TODO: Kijken of het handig kan zijn om de maand erbij te zetten
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.dayofweek

        # Prepare feature set and labels
        feature_columns = [f'power_usage_lag_{i}' for i in range(1, 26)] + ['hour', 'day']
        X = df[feature_columns]
        y = df['appliance_in_use']
        return X, y

    def train(self, data):
        print("Preprocessing data...")
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Fitting model...")
        self.model.fit(X_train, y_train)
        print("Predicting...")
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    def predict(self, power_usages, timestamp):
        if len(power_usages) < 25:
            raise ValueError("At least 25 previous power usage values are required for prediction.")

        timestamp = pd.to_datetime(timestamp)
        hour = timestamp.hour
        day = timestamp.dayofweek

        # Prepare the feature vector
        features = power_usages[-25:] + [hour, day]
        features = np.array(features).reshape(1, -1)

        prediction = self.model.predict(features)
        return prediction[0]

    @classmethod
    def get_model(cls):
        return RandomForestClassifier(verbose=2, n_jobs=8, n_estimators=1000)

    def save_model(self, path):
        # Save the trained model
        joblib.dump(self.model, path)

    @classmethod
    def load_model(cls, path):
        model = joblib.load(path)
        return BasicModel(model=model)

    def visualize(self):
        feature_importances = self.model.feature_importances_
        features = X.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances')
        plt.gca().invert_yaxis()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Sample data preparation (this should be replaced with your actual data loading)


    # Load the model from the model path
    model = BasicModel.load_model(model_path)
    # Example prediction
    # TODO: Data uit de database halen
    example_power_usages_without_microwave = [
        341.03,
        342.36,
        342.52,
        342.07,
        341.77,
        341.66,
        341.84000000000003,
        340.9,
        345.2,
        341.99,
        342.34000000000003,
        346.46,
        340.33,
        345.61,
        345.29,
        345.34,
        343.98,
        344.97,
        344.64,
        343.07,
        344.46,
        346.85,
        346.63,
        345.75,
        346.23
    ]

    timestamp = '2011-04-18 13:23:47'
    print(f"Appliance in use: {model.predict(example_power_usages_without_microwave, timestamp)}")

    example_power_usages_with_microwave = [
        1989.73,
        1978.57,
        1984.44,
        1984.44,
        1995.27,
        1977.32,
        1977.32,
        1977.32,
        1982.53,
        1991.36,
        1995.85,
        1995.85,
        1995.85,
        1995.85,
        2007.7,
        1989.53,
        1968.71,
        1971.8,
        1971.11,
        1977.74,
        1977.74,
        1977.74,
        1977.74,
        1977.74,
        1977.74,
        1977.74,
        1977.74,
        1977.74,
        1977.74,
        1977.74,
    ]

    timestamp = '2011-04-18 14:27:59'
    print(f"Appliance in use: {model.predict(example_power_usages_with_microwave, timestamp)}")

    # model.visualize()
