import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Paths to dataset directories
train_dir = "../Supervised/dataset/datasetsplitted/train"
val_dir = "../Supervised/dataset/datasetsplitted/val"
test_dir = "../Supervised/dataset/datasetsplitted/test"
# Parameters
IMG_SIZE = (64, 64)  # Use a smaller size for training efficiency
BATCH_SIZE = 32
EPOCHS = 20  # Increase for better performance
# Load datasets
train_dataset = image_dataset_from_directory(train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_dataset = image_dataset_from_directory(val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
test_dataset = image_dataset_from_directory(test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
# Get class names and encode labels
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

label_encoder = LabelEncoder()
label_encoder.fit(class_names)
# Normalize dataset
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Define a custom CNN model with explicit Input
def create_cnn():
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)  # Define functional model properly
    return model

cnn_model = create_cnn()
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Stop if validation loss doesn't improve
    patience=5,          # Wait 5 epochs before stopping
    restore_best_weights=True  # Restore the best model
)

# Train CNN with early stopping
history = cnn_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping]  # Add early stopping here
)

# Extract Features from CNN
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)
def extract_features(dataset, feature_extractor):
    features, labels = [], []
    for images, lbls in dataset:
        features.append(feature_extractor.predict(images))
        labels.extend(lbls.numpy())
    return np.vstack(features), np.array(labels)

# Extract features for XGBoost
X_train, y_train = extract_features(train_dataset, feature_extractor)
X_val, y_val = extract_features(val_dataset, feature_extractor)
X_test, y_test = extract_features(test_dataset, feature_extractor)

# Train XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, eval_metric='mlogloss', max_depth=7, learning_rate=0.1)

xgb_model.fit(
    X_train, y_train, 
    eval_set=[(X_val, y_val)],  # Monitor validation performance
    verbose=True
)

# Evaluate on test set
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred, target_names=class_names))

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("CNN Training Performance")
plt.show()
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Compute Hamming Distance
hamming_distance = np.mean(y_pred != y_test)
print(f"Hamming Distance: {hamming_distance}")

from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Get probability scores for each class
y_scores = xgb_model.predict_proba(X_test)

# Plot PR Curve for each class
plt.figure(figsize=(8, 6))

for i in range(len(class_names)):  
    precision, recall, _ = precision_recall_curve(y_test == i, y_scores[:, i])
    auc_score = auc(recall, precision)  # Compute AUC-PR
    
    plt.plot(recall, precision, label=f'Class {class_names[i]} (AUC={auc_score:.2f})')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Each Class")
plt.legend()
plt.grid()
plt.show()
