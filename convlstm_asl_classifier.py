import os, ssl, cv2
import numpy as np
import tensorflow as tf
import neptune
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (ConvLSTM2D, BatchNormalization, MaxPooling3D,
                                     GlobalAveragePooling3D, Dense, Dropout)
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Orthogonal

ssl._create_default_https_context = ssl._create_unverified_context
neptune_api_token = open("./neptune_api_token.txt").read()
run = neptune.init_run(
    project="AmericanSignLanguageRecognition/DynamicLetters",
    api_token=neptune_api_token,
    capture_hardware_metrics=True
)
run["status"] = "running"
run["model/name"] = "ConvLSTM2D-Custom"

image_size = 224
sequence_length = 10
base_path = "./Output"
epochs = 20
batch_size = 2

letters_used = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
label_dict = {letter: idx for idx, letter in enumerate(letters_used)}

def extract_raw_sequences():
    X, y = [], []
    for letter in letters_used:
        folder_path = os.path.join(base_path, letter)
        video_dict = {}
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".jpg"):
                video_id = filename.split("-img-")[0]
                video_dict.setdefault(video_id, []).append(os.path.join(folder_path, filename))

        for frame_paths in video_dict.values():
            if len(frame_paths) >= sequence_length:
                frame_paths = sorted(frame_paths, key=lambda x: int(x.split("-img-")[-1].split(".")[0]))
                sequence = []
                for f in frame_paths[:sequence_length]:
                    img = cv2.resize(cv2.imread(f), (image_size, image_size))
                    img = img.astype(np.float32) / 255.0 #normalization part
                    sequence.append(img)
                X.append(sequence)
                y.append(label_dict[letter])
    return np.array(X), to_categorical(np.array(y), num_classes=len(letters_used))

X, y = extract_raw_sequences()
y_labels = np.argmax(y, axis=1)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in split.split(X, y_labels):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
# LRCN 
# model = Sequential([
#     TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(sequence_length, image_size, image_size, 3)),
#     TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
#     TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
#     TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
#     TimeDistributed(Flatten()),
#     LSTM(64),
#     Dropout(0.5),
#     Dense(len(letters_used), activation='softmax')
# ])


# ConvLSTM2D 
model = Sequential([
    ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', return_sequences=True,
               kernel_initializer=Orthogonal(), input_shape=(sequence_length, image_size, image_size, 3)),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1, 2, 2)),

    ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', return_sequences=True,
               kernel_initializer=Orthogonal()),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1, 2, 2)),

    GlobalAveragePooling3D(),
    Dense(64, activation='relu'),
    Dropout(0.3), # play with this param
    Dense(len(letters_used), activation='softmax')
])
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy']) # add learning rate,
model.summary()


class NeptuneLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        run["train/loss"].log(logs["loss"])
        run["train/accuracy"].log(logs["accuracy"])
        run["val/loss"].log(logs["val_loss"])
        run["val/accuracy"].log(logs["val_accuracy"])

early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=epochs, batch_size=batch_size, callbacks=[early_stop, NeptuneLogger()])
model.save("conv_lstm_custom.h5")
run["artifacts/model_file"].upload("conv_lstm_custom.h5")

def evaluate_model():
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    y_true = np.argmax(y_val, axis=1)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=list(label_dict.keys()),
                                            cmap=plt.cm.Blues, normalize='true')
    plt.title("Normalized Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs("model_trainer", exist_ok=True)
    plt.savefig("model_trainer/conf_matrix_convlstm_custom.png")
    run["eval/conf_matrix"].upload("model_trainer/conf_matrix_convlstm_custom.png")
    report = classification_report(y_true, y_pred, target_names=list(label_dict.keys()))
    print("\n" + report)
    run["eval/classification_report"] = report

evaluate_model()
run["status"] = "completed"
run.stop()
