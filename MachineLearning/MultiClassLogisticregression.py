import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


import app

tf.config.set_visible_devices([], 'GPU')

# Load the data
train_data = pd.read_csv(app.__ROOT__ + '/data/multiclass/train_data.csv')
test_data = pd.read_csv(app.__ROOT__ + '/data/multiclass/test_data.csv')

# Select features and labels
X_train = train_data[['power_usage', 'weekday', 'hour', 'minute']]
y_train = train_data['appliances_in_use']
X_test = test_data[['power_usage', 'weekday', 'hour', 'minute']]
y_test = test_data['appliances_in_use']

# Convert the appliance lists to multi-hot encoded format
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train.apply(eval))  # Apply eval to convert strings to lists
y_test = mlb.transform(y_test.apply(eval))

# Normalize power_usage if necessary
# X_train['power_usage'] = (X_train['power_usage'] - X_train['power_usage'].mean()) / X_train['power_usage'].std()
# X_test['power_usage'] = (X_test['power_usage'] - X_test['power_usage'].mean()) / X_test['power_usage'].std()

# Define the model
model = Sequential([
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(mlb.classes_), activation='sigmoid')  # Use sigmoid for multi-label classification
])

# Compile the model
# model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
# history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])  # Reduced batch size

# Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Save the model
# model.save('multiclass_appliance_model.h5')

# To load the model later, use:
model = tf.keras.models.load_model('multiclass_appliance_model.h5')

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
y_pred = mlb.inverse_transform(y_pred)  # Convert binary predictions to appliance lists
y_test = mlb.inverse_transform(y_test)  # Convert binary labels to appliance lists

# Calculate accuracy
correct = 0
total = 0
for pred, actual in zip(y_pred, y_test):
    if pred == actual:
        correct += 1
    total += 1

accuracy = correct / total

print(f'Test Accuracy: {accuracy}')
