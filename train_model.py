"""
==============================================================
  STEP 2 - MODEL TRAINING SCRIPT
  File: train_model.py
  
  What this does:
  - Loads preprocessed CICIDS2017 data
  - Builds a multi-layer LSTM neural network
  - Trains with early stopping & checkpointing
  - Evaluates model performance
  - Saves model to models/ids_lstm_model.h5
==============================================================
"""

import numpy as np
import os
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_FOLDER  = "models/"
EPOCHS        = 30          # Max epochs (early stopping may stop earlier)
BATCH_SIZE    = 512         # Larger batch = faster training
PATIENCE      = 5           # Stop if no improvement for 5 epochs

os.makedirs(MODEL_FOLDER, exist_ok=True)


def load_data():
    """Load preprocessed numpy arrays."""
    print("[INFO] Loading preprocessed data...")
    X_train = np.load(os.path.join(MODEL_FOLDER, "X_train.npy"))
    X_test  = np.load(os.path.join(MODEL_FOLDER, "X_test.npy"))
    y_train = np.load(os.path.join(MODEL_FOLDER, "y_train.npy"))
    y_test  = np.load(os.path.join(MODEL_FOLDER, "y_test.npy"))
    le      = joblib.load(os.path.join(MODEL_FOLDER, "label_encoder.pkl"))

    print(f"  X_train shape : {X_train.shape}")
    print(f"  X_test shape  : {X_test.shape}")
    print(f"  Classes       : {list(le.classes_)}")
    return X_train, X_test, y_train, y_test, le


def build_lstm_model(n_features, n_classes):
    """
    Build Bidirectional LSTM model.
    
    Architecture:
    Input → BiLSTM(128) → Dropout → BiLSTM(64) → Dropout 
    → BatchNorm → Dense(64) → Dropout → Output(softmax)
    """
    model = Sequential([
        # Layer 1: Bidirectional LSTM captures forward + backward patterns
        Bidirectional(
            LSTM(128, return_sequences=True, activation="tanh"),
            input_shape=(1, n_features)
        ),
        Dropout(0.3),

        # Layer 2: Another LSTM layer for deeper pattern learning
        Bidirectional(
            LSTM(64, return_sequences=False, activation="tanh")
        ),
        Dropout(0.3),

        # Batch normalization for stable training
        BatchNormalization(),

        # Dense classification head
        Dense(64, activation="relu"),
        Dropout(0.2),

        # Output layer: one neuron per attack class
        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_training_history(history):
    """Save training accuracy and loss plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LSTM Training History - CICIDS2017", fontsize=14, fontweight="bold")

    # Accuracy plot
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy", color="#2196F3", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",   color="#FF5722", linewidth=2)
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(history.history["loss"],     label="Train Loss", color="#4CAF50", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   color="#F44336", linewidth=2)
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(MODEL_FOLDER, "training_history.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(cm, classes):
    """Save confusion matrix heatmap."""
    import seaborn as sns
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix - IDS LSTM Model", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_path = os.path.join(MODEL_FOLDER, "confusion_matrix.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def train():
    """Main training pipeline."""
    print("=" * 60)
    print("   LSTM MODEL TRAINING - CICIDS2017 IDS")
    print("=" * 60)

    # 1. Load data
    X_train, X_test, y_train, y_test, le = load_data()
    n_classes  = len(le.classes_)
    n_features = X_train.shape[2]

    # 2. One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes=n_classes)
    y_test_cat  = to_categorical(y_test,  num_classes=n_classes)

    # 3. Build model
    print("\n[INFO] Building LSTM model...")
    model = build_lstm_model(n_features, n_classes)
    model.summary()

    # 4. Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODEL_FOLDER, "ids_lstm_model.h5"),
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1
        ),
    ]

    # 5. Train
    print(f"\n[INFO] Training for max {EPOCHS} epochs (batch size={BATCH_SIZE})...")
    print("  Early stopping will halt training if no improvement.")
    history = model.fit(
        X_train, y_train_cat,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. Evaluate
    print("\n[INFO] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")

    # 7. Classification report
    y_pred  = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 8. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # 9. Save plots
    print("\n[INFO] Saving plots...")
    plot_training_history(history)
    plot_confusion_matrix(cm, le.classes_)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  Model saved : models/ids_lstm_model.h5")
    print(f"  Accuracy    : {test_acc * 100:.2f}%")
    print("=" * 60)
    print("\n  Next step: Run  python app.py")


if __name__ == "__main__":
    train()
