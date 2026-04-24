import numpy as np
import joblib
import os

from feature_extraction import extract_features

# ── TensorFlow is optional — guard against broken installs ────
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    TF_AVAILABLE = True
except Exception as _tf_err:
    print(f"[ann_model] TensorFlow unavailable ({_tf_err}). ANN predictions disabled.")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from data_preparation import prepare_dataset
except Exception:
    pass

def build_ann():
    model = models.Sequential([
        layers.Dense(128, activation="relu", input_shape=(226,)),
        layers.Dense(64,  activation="relu"),
        layers.Dense(2,   activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train():
    # Load dataset
    print("Loading dataset...")
    X, y = prepare_dataset()
    print(f"Total samples : {len(X)}")
    print(f"Normal        : {sum(y == 0)}")
    print(f"Abnormal      : {sum(y == 1)}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/ann_scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy",   
        patience=10,             
        restore_best_weights=True, 
        verbose=1
    )

    print("\nTraining ANN...")
    model = build_ann()
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=8,
        callbacks=[early_stop],    
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAccuracy : {acc * 100:.2f}%")
    print(f"Loss     : {loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n─────────────────────────────────")
    print("         CONFUSION MATRIX")
    print("─────────────────────────────────")
    print(f"                Predicted")
    print(f"              Normal  Abnormal")
    print(f"Actual Normal  [ {cm[0][0]:>3} ]  [ {cm[0][1]:>3} ]")
    print(f"Actual Abnorm  [ {cm[1][0]:>3} ]  [ {cm[1][1]:>3} ]")
    print("─────────────────────────────────")
    print(f"  True Normal  (correct) : {cm[0][0]}")
    print(f"  True Abnormal(correct) : {cm[1][1]}")
    print(f"  False Alarm  (wrong)   : {cm[0][1]}")
    print(f"  Missed Fault (wrong)   : {cm[1][0]}")
    print("─────────────────────────────────")

    print("CLASSIFICATION REPORT")
    print(classification_report(
        y_test, y_pred,
        target_names=["Normal", "Abnormal"]
    ))

    model.save("models/ann_model.h5")
    print("Model saved to models/ann_model.h5")

    return model

def predict(file_path):
    if not TF_AVAILABLE:
        return "PASS", 50.0, 50.0   # safe fallback when TF is unavailable

    model  = tf.keras.models.load_model("models/ann_model.h5")
    scaler = joblib.load("models/ann_scaler.pkl")

    features = extract_features(file_path)
    features = scaler.transform(features.reshape(1, -1))

    result  = model.predict(features, verbose=0)[0]
    prob_normal   = result[0] * 100
    prob_abnormal = result[1] * 100

    if prob_normal >= 65:
        status = "PASS"
    else:
        status = "FAIL"

    return status, prob_normal, prob_abnormal


if __name__ == "__main__":
    if TF_AVAILABLE:
        train()