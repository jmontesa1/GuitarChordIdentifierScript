import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sounddevice as sd
import time
from collections import deque, Counter
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

class GuitarChordRecognizer:
    def __init__(self, model_path=None):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.sample_rate = 22050
        self.audio_window = 2.0
        self.hop_length = 512
        self.n_mels = 128
        self.fmax = 8000 
        self.input_shape = (self.n_mels, 87, 1)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def create_spectrogram(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            y_harmonic = librosa.effects.harmonic(y)
            
            target_length = int(self.audio_window * sr)
            if len(y_harmonic) < target_length:
                y_harmonic = np.pad(y_harmonic, (0, max(0, target_length - len(y_harmonic))))
            else:
                y_harmonic = y_harmonic[:target_length]
            
            S = librosa.feature.melspectrogram(y=y_harmonic, sr=sr,
                                             n_mels=self.n_mels,
                                             hop_length=self.hop_length,
                                             fmax=self.fmax)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            S_dB = (S_dB - np.median(S_dB)) / (np.percentile(S_dB, 95) - np.percentile(S_dB, 5))
            return np.expand_dims(S_dB, axis=-1)
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None
    
    def create_spectrogram_from_buffer(self, y):
        try:
            y_harmonic = librosa.effects.harmonic(y)
            target_length = int(self.audio_window * self.sample_rate)
            
            if len(y_harmonic) < target_length:
                y_harmonic = np.pad(y_harmonic, (0, max(0, target_length - len(y_harmonic))))
            else:
                y_harmonic = y_harmonic[:target_length]
            
            S = librosa.feature.melspectrogram(y=y_harmonic, sr=self.sample_rate,
                                            n_mels=self.n_mels,
                                            hop_length=self.hop_length,
                                            fmax=self.fmax)  
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            S_dB = (S_dB - np.median(S_dB)) / (np.percentile(S_dB, 95) - np.percentile(S_dB, 5))
            return np.expand_dims(S_dB, axis=-1)
        except Exception as e:
            print(f"Error processing audio buffer: {str(e)}")
            return None
    
    def verify_harmonics(self, spectrogram):
        S = librosa.db_to_power(spectrogram[:,:,0])
        
        mel_freqs = librosa.mel_frequencies(n_mels=self.n_mels, fmax=self.fmax)
        
        low_bands = (mel_freqs >= 80) & (mel_freqs < 400)
        mid_bands = (mel_freqs >= 400) & (mel_freqs < 1200)
        high_bands = (mel_freqs >= 1200) & (mel_freqs < self.fmax)
        
        S_avg = np.mean(S, axis=1)
        
        energy = {
            'low': np.mean(S_avg[low_bands]),
            'mid': np.mean(S_avg[mid_bands]),
            'high': np.mean(S_avg[high_bands])
        }
        
        return energy['high'] > 0.2 * energy['low']
    
    def build_model(self, num_classes):
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        return model
    
    def train(self, dataset_dir, epochs=100, batch_size=32):
        spectrograms = []
        labels = []
        
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    chord_label = os.path.basename(root)
                    audio_path = os.path.join(root, file)
                    
                    spectrogram = self.create_spectrogram(audio_path)
                    if spectrogram is not None and self.verify_harmonics(spectrogram):
                        spectrograms.append(spectrogram)
                        labels.append(chord_label)
                        print(f"Processed {file} as {chord_label}")
        if not spectrograms:
            raise ValueError("No valid audio files found")
        
        y = self.label_encoder.fit_transform(labels)
        y = to_categorical(y)
        X = np.array(spectrograms)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = self.build_model(num_classes=len(self.label_encoder.classes_))
        
        print("\nModel summary:")
        self.model.summary()
        
        print("\nTraining model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_acc:.2f}")
    
    def save_model(self, path):
        import joblib
        
        if not path.endswith(('.keras', '.h5')):
            path += '.keras'
        
        self.model.save(path)
        
        meta_path = path + "_meta.joblib"
        joblib.dump({
            'label_encoder': self.label_encoder,
            'config': {
                'sample_rate': self.sample_rate,
                'audio_window': self.audio_window,
                'hop_length': self.hop_length,
                'n_mels': self.n_mels,
                'fmax': self.fmax
            }
        }, meta_path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        import joblib
        
        if not os.path.exists(path):
            if os.path.exists(path + '.keras'):
                path += '.keras'
            elif os.path.exists(path + '.h5'):
                path += '.h5'
            else:
                raise FileNotFoundError(f"No model found at {path}")
        
        self.model = tf.keras.models.load_model(path)
        
        meta_path = path + "_meta.joblib"
        if not os.path.exists(meta_path):
            if path.endswith(('.keras', '.h5')):
                meta_path = path[:-5] + "_meta.joblib"
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        
        meta = joblib.load(meta_path)
        self.label_encoder = meta['label_encoder']
        config = meta['config']
        self.sample_rate = config['sample_rate']
        self.audio_window = config['audio_window']
        self.hop_length = config['hop_length']
        self.n_mels = config['n_mels']
        self.fmax = config.get('fmax', 8000)
        self.input_shape = (self.n_mels, 87, 1)
        print(f"Model loaded from {path}")
    
    def predict_chord(self, audio_data):
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if len(audio_data.shape) > 1:
            audio_data = librosa.to_mono(audio_data)
        
        spectrogram = self.create_spectrogram_from_buffer(audio_data)
        if spectrogram is None or not self.verify_harmonics(spectrogram):
            return None, 0.0
        
        predictions = self.model.predict(np.array([spectrogram]), verbose=0)
        chord_idx = np.argmax(predictions[0])
        return self.label_encoder.classes_[chord_idx], float(np.max(predictions[0]))
    
    def listen_and_recognize(self, smoothing_window=7, device_id=None, debug_vis=False):
        device_id = device_id

        last_print_time = time.time()
        prediction_buffer = deque(maxlen=smoothing_window)
        last_chord = None

        def audio_callback(indata, frames, time_info, status):
            nonlocal last_print_time, last_chord
            
            if status:
                print(f"Audio status: {status}")
            
            mono_audio = indata[:, 0] if indata.ndim > 1 else indata
            rms = np.sqrt(np.mean(mono_audio**2))
            
            current_time = time.time()
            if current_time - last_print_time > 0.1:
                level = min(int(rms * 30), 30)
                print(f"Mic Level: [{'▒'*level}{' '*(30-level)}]", end='\r')
                last_print_time = current_time
            
            if rms > 0.02:
                try:
                    spectrogram = self.create_spectrogram_from_buffer(mono_audio)
                    if spectrogram is None:
                        return

                    if not self.verify_harmonics(spectrogram):
                        return
                    
                    chord, confidence = self.predict_chord(mono_audio)
                    if chord is None:
                        return
                    
                    if confidence > 0.75 or chord != last_chord:
                        prediction_buffer.append((chord, confidence))
                        last_chord = chord
                    
                    if len(prediction_buffer) == smoothing_window:
                        chords, confidences = zip(*prediction_buffer)
                        counts = Counter(chords)
                        most_common = counts.most_common(1)[0]
                        avg_conf = np.mean([conf for c, conf in prediction_buffer 
                                          if c == most_common[0]])
                        
                        print(f"\nDetected: {most_common[0]} "
                             f"(Confidence: {avg_conf:.0%})")
                
                except Exception as e:
                    return

        try:
            print("\nStarting script...")
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=audio_callback,
                blocksize=int(self.sample_rate * 0.5),
                device=device_id
            ):
                while True:
                    time.sleep(0.1)
    
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Audio error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Guitar Chord Recognizer')
    parser.add_argument('--train', help='Path to training dataset directory')
    parser.add_argument('--model', help='Model filename (without extension)', default='guitar_cnn_enhanced')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--live', action='store_true', help='Run live recognition')
    parser.add_argument('--debug', action='store_true', help='Enable visualization')
    parser.add_argument('--device', type=int, help='Audio device ID')
    
    args = parser.parse_args()
    
    if args.train:
        recognizer = GuitarChordRecognizer()
        print(f"Training enhanced model on {args.train}...")
        recognizer.train(args.train, epochs=args.epochs, batch_size=args.batch)
        recognizer.save_model(args.model)
    elif args.live:
        try:
            recognizer = GuitarChordRecognizer(args.model)
        except FileNotFoundError:
            print(f"Model not found at {args.model}. Trying extensions...")
            try:
                recognizer = GuitarChordRecognizer(args.model + '.keras')
            except FileNotFoundError:
                try:
                    recognizer = GuitarChordRecognizer(args.model + '.h5')
                except FileNotFoundError as e:
                    print(f"Failed to load model: {str(e)}")
                    return
        
        try:
            print("\nStarting enhanced recognition...")
            recognizer.listen_and_recognize(
                smoothing_window=7,
                device_id=args.device,
                debug_vis=args.debug
            )
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        return

if __name__ == "__main__":
    main()