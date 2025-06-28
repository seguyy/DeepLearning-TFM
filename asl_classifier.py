#  ASL Letter Classifier with Pretrained CNN Models (Multi-Model Experiment)
import os, ssl, cv2
import numpy as np
import tensorflow as tf
import neptune
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, GlobalAveragePooling2D, RandomRotation, RandomContrast, RandomZoom, RandomFlip)
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import (
    MobileNetV2, DenseNet169, DenseNet201, ResNet50, EfficientNetB0, InceptionV3
)


ssl._create_default_https_context = ssl._create_unverified_context

# Neptune connection
neptune_api_token = open("./neptune_api_token.txt").read()
run = neptune.init_run(
    project="AmericanSignLanguageRecognition/DynamicLetters",
    api_token=neptune_api_token,
    capture_hardware_metrics=True
)
run["status"] = "running"

# Configurations
image_size = 224
#image_size = 299 # for inceptionv3 using another size
sequence_length = 10
base_path = "DynamicLetters_BlackBG/Train"
test_path = "DynamicLetters_BlackBG/Test"
epochs = 50
batch_size = 8
base_model_name = "DenseNet201" # Changing this to try MobileNetV2, ResNet50, etc.

# Detect Labels
letters_used = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
label_dict = {letter: idx for idx, letter in enumerate(letters_used)}

# Load CNN Model based on selection
def load_pretrained_cnn(name):
    if name == "MobileNetV2":
        model = MobileNetV2(include_top=False, input_shape=(image_size, image_size, 3), weights='imagenet')
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif name == "DenseNet169":
        model = DenseNet169(include_top=False, input_shape=(image_size, image_size, 3), weights='imagenet')
        preprocess = tf.keras.applications.densenet.preprocess_input
    elif name == "DenseNet201":
        model = DenseNet201(include_top=False, input_shape=(image_size, image_size, 3), weights='imagenet')
        preprocess = tf.keras.applications.densenet.preprocess_input
    elif name == "ResNet50":
        model = ResNet50(include_top=False, input_shape=(image_size, image_size, 3), weights='imagenet')
        preprocess = tf.keras.applications.resnet50.preprocess_input
    elif name == "EfficientNetB0":
        model = EfficientNetB0(include_top=False, input_shape=(image_size, image_size, 3), weights='imagenet')
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    elif name == "InceptionV3":
        model = InceptionV3(include_top=False, input_shape=(image_size, image_size, 3), weights='imagenet')
        preprocess = tf.keras.applications.inception_v3.preprocess_input
    else:
        raise ValueError("Base model name is invalid.")

    model.trainable = False
    return Sequential([model, GlobalAveragePooling2D()]), preprocess

# Load Model
cnn_model, preprocess_input = load_pretrained_cnn(base_model_name)

# Data Augmentation
data_augmentation = Sequential([
    RandomZoom(0.2),
    RandomContrast(0.1)
])

### Feature Extraction ###

# function below will extract sequences from the folder 
def extract_sequences_by_folder(base_path, label_dict, sequence_length=10):
    X, y, original_sequences = [], [], []

    for letter in label_dict:
        letter_path = os.path.join(base_path, letter)
        for sequence_name in sorted(os.listdir(letter_path)):
            sequence_folder = os.path.join(letter_path, sequence_name)
            frame_files = sorted(
                [f for f in os.listdir(sequence_folder) if f.endswith('.jpg')],
                key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            if len(frame_files) < sequence_length:
                continue

            sequence, restored_seq = [], []
            for f in frame_files[:sequence_length]:
                img_path = os.path.join(sequence_folder, f)
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, (image_size, image_size))

                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                restored_seq.append(img_rgb)

                img_tensor = tf.convert_to_tensor(preprocess_input(img_resized.astype(np.float32)))
                img_tensor = tf.expand_dims(img_tensor, axis=0)
                augmented_tensor = data_augmentation(img_tensor, training=True)
                features = cnn_model.predict(augmented_tensor, verbose=0)[0]
                sequence.append(features)

            X.append(sequence)
            y.append(label_dict[letter])
            original_sequences.append(restored_seq)

    return np.array(X), to_categorical(y, num_classes=len(label_dict)), original_sequences

# Display also misclassified samples
def show_misclassified_examples(X_test, y_test, original_seq_folders, model, label_dict, output_dir="misclassified_examples"):
    os.makedirs(output_dir, exist_ok=True)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    idx_to_label = {v: k for k, v in label_dict.items()}

    for i in range(len(X_test)):
        if y_pred[i] != y_true[i]:
            true_label = idx_to_label[y_true[i]]
            pred_label = idx_to_label[y_pred[i]]

            # Fetch original sequence images
            images = original_seq_folders[i]

            for frame_idx, img in enumerate(images):
                img_name = (
                    f"sequence_{i}_frame_{frame_idx}_true_{true_label}_pred_{pred_label}.png"
                )
                img_path = os.path.join(output_dir, img_name)

                # Save directly using plt.imsave 
                plt.imsave(img_path, img)

                # Upload to Neptune the misclassified one(s)
                run[f"eval/misclassified/{img_name}"].upload(img_path)

label_dict = {"J": 0, "Z": 1}

X_train, y_train, ori_train = extract_sequences_by_folder(
    base_path=base_path,
    label_dict=label_dict,
    sequence_length=sequence_length
)

X_test, y_test, ori_test = extract_sequences_by_folder(
    base_path=test_path,
    label_dict=label_dict,
    sequence_length=sequence_length
)

# LSTM Classifier Setup
input_layer = Input(shape=(sequence_length, cnn_model.output_shape[-1]))
x = LSTM(32, return_sequences=True)(input_layer)
x = Dropout(0.3)(x)
x = LSTM(16)(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.05))(x)
x = Dropout(0.4)(x)
output_layer = Dense(len(letters_used), activation='softmax')(x)
model = Model(input_layer, output_layer)

learning_rate = 0.01  # slightly higher than with Adam, tuning it, can try with 1e-3, 5e-4, 5e-5, etc.
momentum = 0.9
optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)

# Change optimizer to Adam -->> uncomment below
#optimizer = Adam(learning_rate=learning_rate)

# If using Adam -->> uncomment below
# model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy']) # play with the learning rate

model.summary()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# Parameters to log: best model, sequence length, image size and epochs number
run["parameters"] = {
    "base_model": base_model_name,
    "sequence_length": sequence_length,
    "image_size": image_size,
    "epochs": epochs
}

# Early Stopping 
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

class NeptuneLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        run["train/loss"].log(logs["loss"])
        run["train/accuracy"].log(logs["accuracy"])
        run["val/loss"].log(logs["val_loss"])
        run["val/accuracy"].log(logs["val_accuracy"])

#Save the best model
checkpoint = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# Model training
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint, early_stop, NeptuneLogger()]
)

model = tf.keras.models.load_model('best_model.keras')
run["artifacts/best_model"].upload('best_model.keras')
run["parameters/learning_rate"] = learning_rate

# Display misclassified samples
show_misclassified_examples(
    X_test=X_test,
    y_test=y_test,
    original_seq_folders=ori_test,
    model=model,
    label_dict=label_dict
)

# Model Evaluation using confusion matrix
def getConfusionMatrix_numpy():
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=list(label_dict.keys()),
        cmap=plt.cm.Blues, normalize='true'
    )
    disp.ax_.set_title("Normalized Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs("model_trainer", exist_ok=True)
    plt.savefig("model_trainer/confusion_matrix.png")
    run["eval/conf_matrix"].upload("model_trainer/confusion_matrix.png")
    report = classification_report(y_true, y_pred, target_names=list(label_dict.keys()))
    print(report)
    run["eval/classification_report"] = report


getConfusionMatrix_numpy()
run["status"] = "completed"
run.stop()
