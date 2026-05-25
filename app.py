import os
import io
import json
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

# Try lightweight tflite-runtime first, fall back to full tensorflow
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

app = Flask(__name__)

# Reject uploads larger than 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Optional bearer-token auth — set API_TOKEN env var to enable
_API_TOKEN = os.environ.get('API_TOKEN', '')

@app.before_request
def _check_token():
    if not _API_TOKEN:
        return  # auth disabled
    if request.path == '/health':
        return  # always public
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer ') or auth[7:] != _API_TOKEN:
        return jsonify({'error': 'Unauthorized'}), 401

# ── Plant disease model ───────────────────────────────────────────────────────
MODEL_PATH  = os.environ.get('MODEL_PATH',  'plant_disease.tflite')
LABELS_PATH = os.environ.get('LABELS_PATH', 'labels.txt')

disease_interpreter = Interpreter(model_path=MODEL_PATH)
disease_interpreter.allocate_tensors()
disease_input  = disease_interpreter.get_input_details()
disease_output = disease_interpreter.get_output_details()

with open(LABELS_PATH, 'r') as f:
    disease_labels = [line.strip() for line in f if line.strip()]

print(f"Disease model loaded: {MODEL_PATH} | Classes: {len(disease_labels)}")

# ── Sound stress model (optional — server stays up if files are missing) ──────
SOUND_MODEL_PATH  = os.environ.get('SOUND_MODEL_PATH',  'plant_stress_model.tflite')
SOUND_LABELS_PATH = os.environ.get('SOUND_LABELS_PATH', 'sound_labels.json')

sound_interpreter = None
sound_input       = None
sound_output      = None
sound_labels      = {}

try:
    sound_interpreter = Interpreter(model_path=SOUND_MODEL_PATH)
    sound_interpreter.allocate_tensors()
    sound_input  = sound_interpreter.get_input_details()
    sound_output = sound_interpreter.get_output_details()
    with open(SOUND_LABELS_PATH, 'r') as f:
        sound_labels = json.load(f)
    print(f"Sound model loaded: {SOUND_MODEL_PATH} | Classes: {len(sound_labels)}")
except Exception as e:
    print(f"[WARN] Sound model not loaded: {e}. /predict-sound will return 503.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _disease_predict(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    input_data = np.array(image, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    disease_interpreter.set_tensor(disease_input[0]['index'], input_data)
    disease_interpreter.invoke()
    output = disease_interpreter.get_tensor(disease_output[0]['index'])[0]

    top_index = int(np.argmax(output))
    return {
        'label':            disease_labels[top_index],
        'confidence':       float(output[top_index]),
        'all_probabilities': {disease_labels[i]: float(output[i]) for i in range(len(disease_labels))},
    }


def _sound_predict(wav_bytes: bytes) -> dict:
    import librosa

    # Load audio — librosa handles any sample rate and converts to mono
    audio, sr = librosa.load(io.BytesIO(wav_bytes), sr=48000, mono=True, duration=3.0)

    # Pad or trim to exactly 3 s × 48000 = 144000 samples
    target = 144000
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    # Mel spectrogram — identical parameters to the Flutter pipeline
    mel = librosa.feature.melspectrogram(
        y=audio, sr=48000, n_fft=2048, hop_length=256,
        n_mels=128, fmin=0.0, fmax=24000.0,
    )
    mel_db = librosa.power_to_db(mel, top_db=80)

    # Min-max normalise to [0, 1]
    lo, hi = mel_db.min(), mel_db.max()
    norm = (mel_db - lo) / (hi - lo + 1e-9)

    # Apply viridis colormap → RGB uint8 [128, n_frames, 3]
    import matplotlib as _mpl
    viridis = _mpl.colormaps['viridis']
    rgb_full = (viridis(norm)[:, :, :3] * 255).astype(np.uint8)

    # Resize to [128, 128, 3] with PIL bilinear
    img      = Image.fromarray(rgb_full).resize((128, 128), Image.Resampling.BILINEAR)
    tensor   = np.array(img, dtype=np.float32) / 255.0
    tensor   = np.expand_dims(tensor, axis=0)  # [1, 128, 128, 3]

    sound_interpreter.set_tensor(sound_input[0]['index'], tensor)
    sound_interpreter.invoke()
    probs = sound_interpreter.get_tensor(sound_output[0]['index'])[0]

    top_index = int(np.argmax(probs))
    label     = sound_labels.get(str(top_index), sound_labels.get(top_index, 'unknown'))

    return {
        'label':      label,
        'confidence': float(probs[top_index]),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':          'ok',
        'disease_classes':  len(disease_labels),
        'sound_available':  sound_interpreter is not None,
        'sound_classes':    len(sound_labels),
    })


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    try:
        result = _disease_predict(request.files['image'].read())
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict-sound', methods=['POST'])
def predict_sound():
    if sound_interpreter is None:
        return jsonify({'error': 'Sound model not available on this server'}), 503
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    try:
        result = _sound_predict(request.files['audio'].read())
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
