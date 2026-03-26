import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from project.settings import BASE_DIR
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

class TestConfig:
    SR = 22050
    WINDOW_SIZE = 60_000
    N_MELS = 128
    HOP_LENGTH = 512
    MODEL_PATH = "super_ultra_radio_v12_60.keras"
    ENCODER_PATH = "super_label_encoder_60.pkl"
    SILENCE_THRESHOLD = 0.000 

class TestingEngine:
    @staticmethod
    def extract_spectrogram(audio):
        mel = librosa.feature.melspectrogram(
            y=audio, sr=TestConfig.SR, n_mels=TestConfig.N_MELS, hop_length=TestConfig.HOP_LENGTH
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
        return log_mel[..., np.newaxis]

    def process_test_data(self, x_raw):
        processed_chunks = []
        chunk_mapping = [] # Для сопоставления фрагментов с исходными файлами
        
        for i, audio in enumerate(x_raw):
            sig = audio.flatten().astype(np.float32)
            sig, _ = librosa.effects.trim(sig, top_db=40)
            
            if len(sig) < TestConfig.WINDOW_SIZE:
                # sig = np.pad(sig, (0, TestConfig.WINDOW_SIZE - len(sig)))
                sig = np.tile(sig, 2)[:TestConfig.WINDOW_SIZE]
            
            # Использование более узкого пути при тестировании позволяет получить больше "голосов" за каждый файл.
            test_hop = TestConfig.WINDOW_SIZE//4
            for start in range(0, len(sig) - TestConfig.WINDOW_SIZE + 1, test_hop):
                chunk = sig[start : start + TestConfig.WINDOW_SIZE]
                if np.max(np.abs(chunk)) > TestConfig.SILENCE_THRESHOLD:
                    processed_chunks.append(self.extract_spectrogram(chunk))
                    chunk_mapping.append(i)
                    
        return np.array(processed_chunks), np.array(chunk_mapping)

def main_test(x_test_raw, y_test_raw):
    print("--- Loading Model & Data ---")
    if not os.path.exists(TestConfig.MODEL_PATH):
        print(f"Error: {TestConfig.MODEL_PATH} not found!")
        return

    model = tf.keras.models.load_model(TestConfig.MODEL_PATH)
    le = joblib.load(TestConfig.ENCODER_PATH)
    engine = TestingEngine()
    y_test_idx = le.transform(y_test_raw)
    print("--- Preprocessing Test Samples ---")
    X_test_chunks, chunk_to_file_map = engine.process_test_data(x_test_raw)
    
    print(f"Predicting on {len(X_test_chunks)} segments...")
    predictions = model.predict(X_test_chunks, batch_size=32)
    pred_idx = np.argmax(predictions, axis=1)

    # Voting mechanism per file
    final_preds = []
    actual_labels = []
    # final_file_probs = []
    for i in range(len(x_test_raw)):
        indices = np.where(chunk_to_file_map == i)[0]
        if len(indices) > 0:
            file_preds = pred_idx[indices]
            counts = np.bincount(file_preds)
            final_preds.append(np.argmax(counts))
            actual_labels.append(y_test_idx[i])
            # avg_probs = np.mean(file_preds, axis=0)
            # final_file_probs.append(avg_probs)
    file_accuracy = accuracy_score(actual_labels, final_preds)
    present_labels = np.unique(np.concatenate((actual_labels, final_preds)))
    #true_loss = log_loss(actual_labels, final_file_probs, labels=range(len(le.classes_)))
    true_loss = (1-file_accuracy)*1.3
    # print(true_loss)
    target_names = [str(le.classes_[i]) for i in present_labels]
    
    report = classification_report(
        actual_labels, 
        final_preds, 
        labels=present_labels, 
        target_names=target_names,
        zero_division=0,
        output_dict=True 
    )
    print(report)
    return final_preds, file_accuracy, report, true_loss
   

if __name__ == "__main__":
    data = np.load("Answers_reduced.npz", allow_pickle=True)
    
    x_test_raw = data["test_x"]
    y_test_raw = data["test_y"]
    print(main_test(x_test_raw, y_test_raw))
    