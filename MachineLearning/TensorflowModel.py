import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from core.MachineLearningModel import MachineLearningModel


class TensorflowModel(MachineLearningModel):
    def __init__(self, print_progress=True, input_size=5, hidden_size=None):
        super().__init__(print_progress)
        if hidden_size is None:
            hidden_size = [64, 32, 16]

        self.scaler = StandardScaler()

        self.model = keras.Sequential([
            layers.Dense(hidden_size[0], activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(hidden_size[1], activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(hidden_size[2], activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mae', 'mse'])

    def preprocess_data(self, data):
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Extract additional time features
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.dayofweek
        df['weekday'] = df['datetime'].dt.weekday

        # Combine all features
        feature_columns = ['power_usage', 'day', 'hour', 'weekday']
        X = df[feature_columns]
        X = self.scaler.fit_transform(X)
        y = df['appliance_in_use'].values.reshape(-1, 1)  # Reshape for binary classification
        return X, y

    def train(self, data, epochs=100, test_size=0.2, random_state=42):
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1 - test_size,
                                                            test_size=test_size, random_state=random_state)

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

    def predict(self, data):
        X, _ = self.preprocess_data(data)
        y_pred_prob = self.model.predict(X)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten().tolist()
        power_usage_proba = ((y_pred_prob / 5) + 1).flatten().tolist()  # Adjust for power usage probability
        return y_pred, power_usage_proba

    def save_model(self, model):
        self.model.save(model)
