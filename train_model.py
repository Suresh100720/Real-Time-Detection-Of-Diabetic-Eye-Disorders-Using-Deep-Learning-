"""Train a modern Keras CNN from the project's supplied train_data.npy file."""
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "train_data.npy"
MODEL_PATH = ROOT / "models" / "eye_disease.keras"
CLASS_NAMES = ["Cataract", "Glaucoma", "Mild diabetic retinopathy", "Moderate diabetic retinopathy", "Normal", "Proliferative diabetic retinopathy", "Diabetic eye disease", "Severe diabetic retinopathy"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("train_data.npy is missing. Keep it in the same folder as this script.")
    print("Loading supplied training data...")
    records = np.load(DATA_PATH, allow_pickle=True)
    images, labels = [], []
    for record in records:
        image = np.asarray(record[0], dtype=np.float32)
        label = np.asarray(record[1], dtype=np.float32)
        if image.shape == (50, 50, 3) and label.shape == (8,):
            images.append(image / 255.0)
            labels.append(int(np.argmax(label)))
    if len(images) < 10:
        raise ValueError("Not enough valid labeled images were found in train_data.npy.")
    x = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model = keras.Sequential([
        layers.Input(shape=(50, 50, 3)),
        layers.Conv2D(32, 3, activation="relu", padding="same"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(128, activation="relu"), layers.Dropout(0.35),
        layers.Dense(8, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=15, batch_size=32, callbacks=callbacks)
    model.save(MODEL_PATH)
    loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Validation accuracy: {accuracy:.3f} (evaluation on this supplied dataset only)")
    print("Class order:", ", ".join(f"{i}: {name}" for i, name in enumerate(CLASS_NAMES)))

if __name__ == "__main__":
    main()
