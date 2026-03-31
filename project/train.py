import os
import random
import hashlib as h
import numpy as np
import librosa
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, callbacks, optimizers


class Config:
    SR = 22050
    WINDOW_SIZE = 60_000
    N_MELS = 128
    HOP_LENGTH = 512
    BATCH_SIZE = 32
    EPOCHS = 120
    LEARNING_RATE = 1e-3
    MODEL_PATH = "super_ultra_radio_v12_60.keras"
    ENCODER_PATH = "super_label_encoder_60.pkl"
    SILENCE_THRESHOLD = 0.005
    SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(Config.SEED)

class AudioEngine:
    @staticmethod
    def recover_labels(data_y, max_classes=50):
        races = []
        for y in labels_data:
            hi = y[:32]
            name = y[32:]
            for race in range(0,50):
                hash = h.md5((str(race)+name).encode()).hexdigest()
                if hash == hi:
                    races.append(race)
        return np.array(races, dtype=np.int32)

    @staticmethod
    def extract_spectrogram(audio):
        mel = librosa.feature.melspectrogram(
            y=audio, sr=Config.SR, n_mels=Config.N_MELS, hop_length=Config.HOP_LENGTH
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
        return log_mel[..., np.newaxis]

    def prepare_dataset(self, x_raw, y_raw, is_training=True):
        X, Y = [], []
        hop = Config.WINDOW_SIZE if is_training else Config.WINDOW_SIZE//3
        
        for i, audio in enumerate(x_raw):
            sig = audio.flatten().astype(np.float32)
            sig, _ = librosa.effects.trim(sig, top_db=40)
            
            if len(sig) < Config.WINDOW_SIZE:
                sig = np.pad(sig, (0, Config.WINDOW_SIZE - len(sig)))
                # sig = np.tile(sig, 2)[:Config.WINDOW_SIZE]
            
            for start in range(0, len(sig) - Config.WINDOW_SIZE + 1, hop):
                chunk = sig[start : start + Config.WINDOW_SIZE]
                if np.max(np.abs(chunk)) > Config.SILENCE_THRESHOLD:
                    X.append(self.extract_spectrogram(chunk))
                    Y.append(y_raw[i])
        return np.array(X), np.array(Y)

def se_block(inputs, ratio=8):
    filters = inputs.shape[-1]
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([inputs, se])

def res_block(x, f):
    shortcut = x
    x = layers.Conv2D(f, 3, padding="same", activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f, 3, padding="same", activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x)
    if shortcut.shape[-1] != f:
        shortcut = layers.Conv2D(f, 1, padding="same")(shortcut)
    return layers.Add()([x, shortcut])

def build_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.BatchNormalization()(inputs)
    # x = layers.GaussianNoise(stddev=0.01)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="swish")(x)
    x = res_block(x, 64)
    x = layers.MaxPooling2D(2)(x)
    
    x = res_block(x, 128)
    x = layers.MaxPooling2D(2)(x)
    
    x = res_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(256, activation="swish")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(Config.LEARNING_RATE), 
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model

def main():
    print("--- Загрузка и восстановление меток")
    data = np.load("Data.npz", allow_pickle=True)
    engine = AudioEngine()
    
    y_tr_raw = engine.recover_labels(data["train_y"])
    y_val_raw = engine.recover_labels(data["valid_y"])
    
    le = LabelEncoder()
    y_tr_idx = le.fit_transform(y_tr_raw)
    y_val_idx = le.transform(y_val_raw)
    num_classes = len(le.classes_)
    joblib.dump(le, Config.ENCODER_PATH)

    print("--- Подготовка признаков")
    X_tr, y_tr_idx = engine.prepare_dataset(data["train_x"], y_tr_idx, is_training=True)
    X_val, y_val_idx = engine.prepare_dataset(data["valid_x"], y_val_idx, is_training=False)

    unique_classes = np.unique(y_tr_idx)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_tr_idx
    )
    class_weight_dict = dict(zip(unique_classes, weights))

    y_tr_oh = tf.keras.utils.to_categorical(y_tr_idx, num_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val_idx, num_classes)

    model = build_model(X_tr.shape[1:], num_classes)
    
    cb = [
        callbacks.ModelCheckpoint(Config.MODEL_PATH, save_best_only=True, monitor="val_accuracy"),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    ]

    model.fit(
        X_tr, y_tr_oh, 
        validation_data=(X_val, y_val_oh),
        batch_size=Config.BATCH_SIZE, 
        epochs=Config.EPOCHS, 
        callbacks=cb,
    )

if __name__ == "__main__":
    main()
# the best of the bests
