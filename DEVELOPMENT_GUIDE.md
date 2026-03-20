# 🔊 Engine Acoustic Diagnostic AI — Complete Development Guide

---

## 📁 PROJECT FILE STRUCTURE

```
engine_ai/                        ← Your main project folder
│
├── requirements.txt              ← Python libraries to install
├── config.py                     ← All settings (sample rate, paths, etc.)
├── feature_extraction.py         ← Converts audio → numbers for the AI
├── data_preparation.py           ← Scans dataset folders, builds training data
├── train_model.py                ← Trains and saves the AI model
├── predict.py                    ← Loads model, predicts, generates plots
├── app.py                        ← Gradio web UI (run this last)
├── test_single_file.py           ← Quick command-line testing tool
│
├── dataset/
│   ├── normal/                   ← Put GOOD engine audio files here
│   └── abnormal/                 ← Put FAULTY engine audio files here
│
└── models/                       ← Auto-created when you train
    ├── engine_model.pkl          ← Saved AI model
    └── scaler.pkl                ← Saved feature scaler
```

---

## ✅ STEP 1 — INSTALL PYTHON AND VS CODE

### Install Python
1. Go to https://www.python.org/downloads/
2. Download **Python 3.11** (click the big yellow button)
3. Run the installer
4. ⚠️ **IMPORTANT**: On the first screen, tick the box **"Add Python to PATH"**
5. Click "Install Now"

### Install VS Code
1. Go to https://code.visualstudio.com/
2. Download and install for your OS (Windows/Mac/Linux)

### Install the Python Extension in VS Code
1. Open VS Code
2. Click the Extensions icon in the left sidebar (looks like 4 squares)
3. Search for **"Python"**
4. Click Install on the one by Microsoft

---

## ✅ STEP 2 — CREATE YOUR PROJECT FOLDER IN VS CODE

1. Open VS Code
2. Go to **File → Open Folder**
3. Create a new folder called `engine_ai` somewhere on your computer (e.g., Desktop)
4. Open that folder in VS Code
5. You will see an empty Explorer panel on the left

---

## ✅ STEP 3 — CREATE ALL THE CODE FILES

In VS Code, create each file below:

**How to create a file:**
- In the left Explorer panel, click the **New File** icon (paper with a + sign)
- Type the filename and press Enter
- Paste the code from each file provided

Files to create:
1. `requirements.txt`
2. `config.py`
3. `feature_extraction.py`
4. `data_preparation.py`
5. `train_model.py`
6. `predict.py`
7. `app.py`
8. `test_single_file.py`

---

## ✅ STEP 4 — OPEN THE TERMINAL IN VS CODE

1. In VS Code, go to **Terminal → New Terminal**
2. A terminal panel opens at the bottom of the screen
3. It should already be in your `engine_ai` folder

**Check Python is working:**
```
python --version
```
You should see something like `Python 3.11.x`

---

## ✅ STEP 5 — INSTALL ALL REQUIRED LIBRARIES

In the VS Code terminal, run this command:

```
pip install -r requirements.txt
```

This will download and install:
- `librosa`    — audio analysis library
- `scikit-learn` — machine learning library
- `gradio`     — web UI library
- `matplotlib` — plotting library
- `numpy`, `pandas`, `joblib`, `soundfile`

⏳ **This may take 3–5 minutes** — wait until it finishes.

---

## ✅ STEP 6 — ADD YOUR AUDIO FILES

### Create the dataset folders
Run this in the terminal:
```
python data_preparation.py
```

This creates:
- `dataset/normal/`     ← Put GOOD engine recordings here
- `dataset/abnormal/`   ← Put FAULTY engine recordings here
- `models/`             ← Will be filled when you train

### Copy your audio files
- **Normal engine sounds** → copy into `dataset/normal/`
- **Abnormal engine sounds** → copy into `dataset/abnormal/`

### Supported formats
`.wav` (best), `.mp3`, `.flac`, `.ogg`

### How many files do you need?
| Files per class | Expected accuracy |
|-----------------|-------------------|
| 5–10            | 70–80% (basic demo) |
| 20–50           | 85–92% (good) |
| 100+            | 93–98% (production) |

---

## ✅ STEP 7 — TRAIN THE AI MODEL

Once your audio files are in place, run:
```
python train_model.py
```

**What happens:**
1. 📁 Scans both dataset folders
2. 🎵 Loads every audio file
3. 🔢 Extracts 226 acoustic features from each file
4. 🌲 Trains a Random Forest AI model
5. 📊 Shows accuracy and evaluation results
6. 💾 Saves the model to `models/engine_model.pkl`

**Example output:**
```
  Normal   recordings found : 30
  Abnormal recordings found : 28
  ...
  Model Accuracy: 94.83%
  ...
  Model saved → models/engine_model.pkl
  Scaler saved → models/scaler.pkl
  Training pipeline complete! You can now run: python app.py
```

---

## ✅ STEP 8 — RUN THE WEB APP

```
python app.py
```

Then open your browser at: **http://localhost:7860**

**You will see the web interface with:**
- Audio upload button (or microphone recording)
- Analyze Engine button
- Diagnostic report (PASS / FAIL + confidence %)
- 4 visual plots: waveform, spectrogram, mel spectrogram, confidence gauge

---

## ✅ STEP 9 — TEST A SINGLE FILE (OPTIONAL)

If you want to test quickly without opening the browser:
```
python test_single_file.py dataset/normal/engine_01.wav
```

Or test an entire folder:
```
python test_single_file.py dataset/abnormal/ abnormal
```

---

## 📊 HOW THE AI WORKS (Simple Explanation)

```
Your Audio File (.wav)
        ↓
  [feature_extraction.py]
  Convert sound → 226 numbers
  (MFCC fingerprint, brightness, energy, pitch patterns)
        ↓
  [train_model.py / predict.py]
  Random Forest AI Model
  (200 decision trees vote on PASS or FAIL)
        ↓
  Result: PASS ✅ or FAIL ❌
  Confidence Score: e.g. 94.3%
        ↓
  [predict.py]
  Generate 4 visualisation plots
        ↓
  [app.py]
  Display in web browser
```

---

## 🔧 COMMON ERRORS AND FIXES

### "Python is not recognized as a command"
→ Reinstall Python and make sure to tick **"Add Python to PATH"** during install.

### "ModuleNotFoundError: No module named 'librosa'"
→ Run: `pip install -r requirements.txt`

### "No audio files found in dataset/normal"
→ Make sure your .wav files are inside the `dataset/normal/` folder (not in a subfolder inside it).

### "Model file not found"
→ You need to train first: `python train_model.py`

### "Error loading audio file"
→ The audio file might be corrupted. Try converting it to .wav format using Audacity (free).

### Port 7860 already in use
→ Change `server_port = 7860` to `server_port = 7861` in `app.py`

---

## 🚀 FUTURE IMPROVEMENTS (After Demo)

Once the demo works, these enhancements are possible:

| Feature | How |
|---------|-----|
| Real-time microphone monitoring | Use `sounddevice` library with continuous recording |
| Detect specific defect type | Add more label classes (knock, rattle, misfire) |
| Better accuracy with more data | Collect 200+ recordings per class |
| Deep learning model | Replace Random Forest with a CNN on spectrograms |
| Factory system integration | Add REST API endpoint using FastAPI |
| Auto-report generation | Export PDF reports with matplotlib + reportlab |

---

## 📚 KEY TERMS GLOSSARY

| Term | Meaning |
|------|---------|
| MFCC | Mel-Frequency Cepstral Coefficients — compact fingerprint of sound texture |
| Spectrogram | Image showing which frequencies are present at each moment in time |
| Mel Scale | Frequency scale that matches how humans hear pitch |
| Random Forest | AI algorithm: 200 decision trees all vote on the answer |
| Feature Extraction | Converting raw audio into numbers the AI model can learn from |
| StandardScaler | Makes all feature numbers the same scale (important for accuracy) |
| Confidence Score | How certain the AI is about its PASS/FAIL decision |
| AUC-ROC | A metric from 0 to 1 measuring model quality (1.0 = perfect) |
