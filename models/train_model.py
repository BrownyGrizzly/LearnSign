from sklearn.model_selection import train_test_split
import tensorflow as tf
keras = tf.keras
to_categorical = keras.utils.to_categorical
Sequential = keras.models.Sequential
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense
load_model = keras.models.load_model
TensorBoard = keras.callbacks.TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np
import os

# Define paths and actions
DATA_PATH = os.path.join('..', 'data', 'actions')
actions = np.array(['hello', 'thanks', 'iloveyou', 'name'])
sequence_length = 30

# Label mapping
label_map = {label: num for num, label in enumerate(actions)}

# Load and prepare data
sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Build and compile model
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])
model.summary()

# Save the model
model.save('action.h5')

# Evaluate the model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))

# Load the model
model = load_model('action.h5')

# Evaluate the model
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))

# Make predictions
res = model.predict(X_test)
predicted_action = actions[np.argmax(res[4])]
true_action = actions[np.argmax(y_test[4])]
print(predicted_action)
print(true_action)
