# 🛡️ AI Fake Job Advertisement Detection & Verification System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RohindhShan/Job_Detection/blob/main/Job_Detection.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.20%2B-ff7c00.svg?logo=gradio&logoColor=white)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end, multimodal artificial intelligence system designed to detect and verify fraudulent job advertisement posters and social media recruitment flyers. 

The system combines **Computer Vision** (a lightweight Convolutional Neural Network trained on visual layout patterns) with **Optical Character Recognition (OCR) & Rule-Based NLP Heuristics** via a **60/40 Hybrid Decision Fusion Engine**, complete with an interactive **Gradio Web Interface**.

---

## 📌 Table of Contents
- [Problem Overview](#-problem-overview)
- [System Architecture & Methodology](#-system-architecture--methodology)
- [Heuristic Red Flag Rules](#-heuristic-red-flag-rules)
- [Project Structure](#-project-structure)
- [Quick Start: Google Colab](#-quick-start-google-colab)
- [Local Installation & Setup](#-local-installation--setup)
- [Running the Application](#-running-the-application)
- [Evaluation & Performance](#-evaluation--performance)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🚨 Problem Overview

Employment scams have surged across LinkedIn, Telegram, WhatsApp, and social media platforms. Fraudulent recruiters exploit job seekers through:
- **Visual Clutter & Typography Chaos**: Gaudy banners, excessive exclamation marks, mismatched fonts, and urgency borders.
- **Advance-Fee Fraud**: Demanding mandatory "registration fees", "training kit deposits", or "laptop dispatch charges".
- **Informal Communication Channels**: Directing applicants exclusively to WhatsApp, personal phone numbers, or anonymous Telegram channels.
- **Unrealistic Compensation**: Promising $500–$1,500/day or ₹50,000/day for minimal, low-skill data entry tasks.
- **Public Domain Email Addresses**: Posing as corporate recruiters while using free `@gmail.com`, `@yahoo.com`, or disposable mail domains.

This project addresses these threats using a multimodal approach: analyzing both visual design anomalies and textual semantic red flags.

---

## 🧠 System Architecture & Methodology

```
                      ┌───────────────────────────────────────────┐
                      │    Uploaded Job Ad Poster / Screenshot    │
                      └─────────────────────┬─────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
         [Computer Vision Stream]                        [Text & NLP Stream]
         - Resized to 128x128                            - 2x Cubic Upscaling & Grayscale
         - Tensor Normalization (1/255)                  - Otsu Adaptive Binarization
         - 4-Block Lightweight CNN                       - Pytesseract OCR Extraction
         - GlobalAveragePooling2D                        - Regex Rule Verification Engine
                    │                                               │
                    ▼                                               ▼
          Visual Scam Probability                          Scam Risk Score &
                 (0 - 100%)                             Itemized Red Flag List
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            ▼
                        ┌───────────────────────────────────────┐
                        │    Multimodal Hybrid Fusion Engine    │
                        │   Risk = (0.60 * CNN) + (0.40 * NLP)  │
                        └───────────────────┬───────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
      Final Risk Score ≥ 50%                          Final Risk Score < 50%
     🚨 FAKE ADVERTISEMENT                           ✅ GENUINE ADVERTISEMENT
```

### 1. Synthetic Dataset Generation (`CELL 2`)
- Programmatically renders balanced training and validation sets (~1,500 posters) via `PIL.ImageDraw`.
- **Genuine Posters**: Corporate blue/slate palettes, formal headers, corporate email domains, realistic compensation ranges, and standard hiring disclaimers.
- **Fake Posters**: Garish yellow/red warning borders, hyperbolic earnings, upfront fee demands, informal messaging contacts, and intentional typographic traps.
- Generates dedicated test samples and compresses them into `test_job_posters_dataset.zip` for instant testing.

### 2. High-Speed Image Processing Pipeline (`CELL 3`)
- Uses vectorized tensor normalization (rescaling to $[0, 1]$) with gentle data augmentation (subtle rotations, slight spatial shifts, and zooms) via Keras `ImageDataGenerator`.
- Operates at $128 \times 128$ resolution for rapid batch processing without CPU bottlenecks.

### 3. Lightweight CNN Classifier (`CELL 4`)
- **Architecture**: 4 convolutional blocks (16, 32, 64, 128 filters with $3 \times 3$ kernels), each paired with `BatchNormalization`, `MaxPooling2D`, and `Dropout(0.2)`.
- **Global Average Pooling**: Replaces parameter-heavy dense flattening layers, dropping total trainable parameters from ~6.8M to ~110k.
- **Training Time**: Trains in under 10 seconds on standard CPUs and converges rapidly with `EarlyStopping` and `ReduceLROnPlateau`.
- Saves in modern native Keras format (`model.keras`) with fallback to legacy (`model.h5`).

### 4. Rule-Based OCR & NLP Engine (`CELL 5`)
- Preprocesses posters with 2x bicubic upscaling and adaptive Otsu thresholding before passing to Pytesseract (`--psm 6`).
- Scans extracted text against a curated dictionary of regular expressions detecting predatory recruiting language.

### 5. Multimodal Decision Fusion (`CELL 5`)
$$\text{Final Risk Score} = (0.60 \times \text{CNN Probability}) + (0.40 \times \text{Rule-Based Risk Score})$$
- A score $\ge 50.0$ triggers a **FAKE** verdict; a score $< 50.0$ classifies the ad as **GENUINE**.

### 6. Interactive Gradio Web Interface (`CELL 6` & `app.py`)
- Provides drag-and-drop poster upload, one-click pre-generated test samples, risk meter breakdown, OCR text preview, and an itemized explanation of every detected red flag.

---

## 🚩 Heuristic Red Flag Rules

| Risk Category | Regex Indicators | Penalty Score | Example Trigger |
| :--- | :--- | :---: | :--- |
| **Upfront Fee / Deposit** | `registration fee`, `security deposit`, `kit fee`, `laptop dispatch fee`, `refundable deposit` | **+40% (Critical)** | *"Security deposit Rs. 1500 for training kit"* |
| **Free Public Email Domain** | `@gmail.com`, `@yahoo.com`, `@hotmail.com`, `@outlook.com`, `@rediffmail.com` | **+30% (High)** | *"career.fastrecruitment@gmail.com"* |
| **Direct / Informal Contact** | `whatsapp only`, `dm on whatsapp`, `telegram hr`, `contact via whatsapp` | **+25% (High)** | *"WhatsApp ONLY: +1 (555) 942-8819"* |
| **Unrealistic Salary Claims** | `earn $500 daily`, `50,000/day guaranteed`, `instant cash`, `daily payout` | **+25% (High)** | *"Earn 50,000/Day Guaranteed!"* |
| **Pressure Tactics & No Interview** | `urgent hiring`, `only 5 seats left`, `no interview`, `no resume`, `instant joining` | **+20% (Suspicious)** | *"NO INTERVIEW! NO RESUME REQUIRED!"* |
| **Typographical Trap Signatures** | `congratualations`, `refundible`, `joing`, `intrview`, `registartion`, `requried` | **+15% (Suspicious)** | *"Registartion fee fully refundible"* |

---

## 📁 Project Structure

```
Job_Detection/
├── Job_Detection.ipynb       # Primary Jupyter Notebook (Colab & local ready)
├── AI_Detection1.ipynb       # Synchronized working notebook
├── app.py                    # Standalone Gradio Web Application script
├── requirements.txt          # Python dependencies
├── .gitignore                # Git exclusions (datasets, heavy models, cache)
└── README.md                 # Project documentation
```

---

## ⚡ Quick Start: Google Colab

The easiest way to run the entire pipeline without local configuration is via Google Colab:

1. Click the **[Open In Colab](https://colab.research.google.com/github/RohindhShan/Job_Detection/blob/main/Job_Detection.ipynb)** badge.
2. Select **Runtime** > **Run all** (or step through cells 1 to 7).
3. In **Cell 6**, launch the interactive Gradio Web UI link directly in your browser!

---

## 💻 Local Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git
- Tesseract OCR engine installed on your system:
  - **Ubuntu / Debian**:
    ```bash
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr libtesseract-dev
    ```
  - **macOS (Homebrew)**:
    ```bash
    brew install tesseract
    ```
  - **Windows**:
    Download and run the installer from [UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki). Ensure the installation path (typically `C:\Program Files\Tesseract-OCR`) is added to your system `PATH`.

### Step 1: Clone the Repository
```bash
git clone https://github.com/RohindhShan/Job_Detection.git
cd Job_Detection
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### Option 1: Standalone Gradio Web App
Run the self-contained web app directly:
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`.

### Option 2: Jupyter Notebook
Launch the notebook in JupyterLab or VS Code:
```bash
jupyter notebook Job_Detection.ipynb
```
Execute cells sequentially to generate the dataset, train the CNN model, inspect confusion matrices, and run the Gradio interface.

---

## 📊 Evaluation & Performance

- **Model Parameters**: ~110,000 parameters (ultra-lightweight footprint).
- **Training Time**: ~8–12 seconds on standard quad-core CPU for 5 epochs.
- **Validation Accuracy**: Converges to $>98\%$ on synthetic layout distinction.
- **Hybrid Performance**:
  - Genuine advertisement validation: accurately recognized without false fee alerts.
  - Scam advertisement validation: flagged with high risk ($\ge 85\%$) detailing exact violation categories.

---

## 🛠️ Tech Stack

- **Deep Learning**: [TensorFlow](https://www.tensorflow.org/) / [Keras 3](https://keras.io/)
- **Computer Vision & OCR**: [OpenCV](https://opencv.org/), [Pillow (PIL)](https://python-pillow.org/), [Pytesseract](https://github.com/madmaze/pytesseract)
- **Web Application**: [Gradio](https://gradio.app/)
- **Data Science & Metrics**: [NumPy](https://numpy.org/), [Scikit-learn](https://scikit-learn.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
