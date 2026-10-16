"""
AI-Based Fake Job Advertisement Detection & Verification System
===============================================================
Hybrid Multimodal Decision Fusion:
- 60% Lightweight CNN (Visual clutter, typography chaos & layout anomalies)
- 40% Pytesseract OCR + Regex Heuristic Rules (Upfront fees, informal contacts, fake domains)
"""

import os
import re
import glob
import shutil
import random
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import pytesseract
import gradio as gr

# Cross-platform Tesseract executable configuration
if os.path.exists('/usr/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
elif shutil.which('tesseract'):
    pytesseract.pytesseract.tesseract_cmd = shutil.which('tesseract')
elif os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
elif os.path.exists(r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_HEIGHT = 128
IMG_WIDTH = 128
TEST_DIR = 'test_samples'

# ==============================================================================
# 1. LIGHTWEIGHT CNN ARCHITECTURE
# ==============================================================================
def build_fake_job_cnn(input_shape=(128, 128, 3)):
    """
    Optimized CNN Architecture for Job Ad Classification:
    Uses Input layer and GlobalAveragePooling (reduces params to ~110k for fast CPU/GPU inference).
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Block 3
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Block 4
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Global Average Pooling + Dense Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="FastFakeJobCNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_or_initialize_model():
    """Load existing model weights or initialize a compiled instance."""
    for model_path in ['model.keras', 'model.h5']:
        if os.path.exists(model_path):
            try:
                print(f">> Loading pretrained model from '{model_path}'...")
                return tf.keras.models.load_model(model_path)
            except Exception as e:
                print(f"[Warning] Could not load '{model_path}': {e}")
    print(">> No existing model file found. Building fresh CNN architecture...")
    return build_fake_job_cnn()

# ==============================================================================
# 2. HEURISTIC RED FLAG PATTERNS & OCR EXTRACTION
# ==============================================================================
PATTERNS = {
    'free_email': re.compile(r'\b[A-Za-z0-9._%+-]+@(gmail|yahoo|hotmail|outlook|rediffmail|live|aol|protonmail|yopmail)\.com\b', re.IGNORECASE),
    'payment_fee': re.compile(r'\b(registration\s*fee|security\s*deposit|training\s*fee|kit\s*fee|refundable\s*fee|dispatch\s*fee|processing\s*fee|equipment\s*fee|verification\s*fee|pay\s*\$\d+|deposit\s*rs|fee\s*\$\d+)\b', re.IGNORECASE),
    'whatsapp_telegram': re.compile(r'\b(whatsapp\s*only|whatsapp\s*me|dm\s*on\s*whatsapp|contact\s*hr\s*on\s*whatsapp|telegram\s*hr|telegram\s*admin|contact\s*via\s*whatsapp|\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b', re.IGNORECASE),
    'crazy_salary': re.compile(r'\b(earn\s*\$?\d+[kK]?\s*\/?\s*(day|daily)|50,?000\s*\/?\s*(day|daily)|instant\s*cash|daily\s*payout|weekly\s*\$\d{3,4}|rs\s*45,?000|work\s*2\s*hrs?\s*earn)\b', re.IGNORECASE),
    'urgency_scam': re.compile(r'\b(urgent\s*hiring|only\s*\d+\s*seats|no\s*interview|no\s*resume|guaranteed\s*selection|immediate\s*joining|apply\s*in\s*\d+\s*mins|apply\s*now|apply\s*instantly|guaranteed\s*job)\b', re.IGNORECASE),
    'spelling_mistakes': re.compile(r'\b(congratualations|refundible|joing|intrview|registartion|requried|immediatley|vacany)\b', re.IGNORECASE)
}

def extract_text_from_image(image_input):
    """
    Extract text using pytesseract OCR with cubic upscaling and adaptive Otsu binarization.
    """
    try:
        if isinstance(image_input, str):
            cv_img = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            cv_img = image_input.copy()
        else: # PIL Image
            cv_img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

        if cv_img is None:
            return ""

        # Grayscale and 2x cubic upscaling for superior character sharpness
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # Otsu thresholding
        _, thresh = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # PSM 6: Assume a single uniform block of text
        extracted_text = pytesseract.image_to_string(thresh, config='--psm 6').strip()
        if len(extracted_text) < 15:
            # Fallback without thresholding
            extracted_text = pytesseract.image_to_string(upscaled).strip()

        return extracted_text
    except Exception as e:
        print(f"[Notice] OCR fallback triggered: {e}")
        return ""

def rule_based_verification(text):
    """
    Rule-Based NLP Verification:
    Calculates scam risk score (0 to 100%) and itemizes red flags found.
    """
    red_flags = []
    rule_score = 0.0

    if not text or len(text.strip()) < 5:
        return {
            'rule_score': 30.0,
            'red_flags': ["Warning: Image contains minimal or unreadable text (vague advertisement layout)."]
        }

    # 1. Free/Disposable Email Domain Check (High Risk: +30%)
    email_matches = PATTERNS['free_email'].findall(text)
    if email_matches:
        rule_score += 30.0
        red_flags.append(f"High Risk: Uses free/unverified public email domain (@{email_matches[0]}.com) instead of corporate domain.")

    # 2. Registration Fee / Upfront Payment (Critical Risk: +40%)
    fee_matches = PATTERNS['payment_fee'].findall(text)
    if fee_matches:
        rule_score += 40.0
        red_flags.append(f"Critical Risk: Demands upfront money/deposit ('{fee_matches[0]}'). Legitimate employers NEVER charge application fees.")

    # 3. WhatsApp / Telegram Direct Messaging Only (+25%)
    contact_matches = PATTERNS['whatsapp_telegram'].findall(text)
    if contact_matches:
        rule_score += 25.0
        red_flags.append(f"High Risk: Direct messaging or informal contact specified ('{contact_matches[0]}').")

    # 4. Unrealistic High Salary Claims (+25%)
    salary_matches = PATTERNS['crazy_salary'].findall(text)
    if salary_matches:
        rule_score += 25.0
        red_flags.append(f"High Risk: Unrealistic earning promise detected ('{salary_matches[0]}').")

    # 5. Pressure / Urgency Tactics & 'No Interview' (+20%)
    urgency_matches = PATTERNS['urgency_scam'].findall(text)
    if urgency_matches:
        rule_score += 20.0
        red_flags.append(f"Suspicious: Artificial urgency or 'no interview' claim ('{urgency_matches[0]}').")

    # 6. Egregious Spelling & Grammar Traps (+15%)
    spelling_matches = PATTERNS['spelling_mistakes'].findall(text)
    if spelling_matches:
        rule_score += 15.0
        red_flags.append(f"Suspicious: Typographical errors common in scam templates ('{spelling_matches[0]}').")

    rule_score = min(rule_score, 100.0)

    if not red_flags:
        red_flags.append("Verified: No common scam keywords detected. Ad uses professional terminology.")

    return {
        'rule_score': rule_score,
        'red_flags': red_flags
    }

def verify_job_advertisement(image_path_or_array, model):
    """
    HYBRID ENSEMBLE DECISION:
    Combines 60% CNN Visual Probability + 40% Rule-Based OCR Risk Score.
    """
    if isinstance(image_path_or_array, str):
        pil_img = Image.open(image_path_or_array).convert('RGB')
    elif isinstance(image_path_or_array, np.ndarray):
        pil_img = Image.fromarray(image_path_or_array).convert('RGB')
    else:
        pil_img = image_path_or_array.convert('RGB')

    img_resized = pil_img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    processed_batch = np.expand_dims(img_array, axis=0)

    # 1. CNN Visual Prediction (Probability of FAKE)
    try:
        cnn_prob = float(model.predict(processed_batch, verbose=0)[0][0]) * 100.0
    except Exception as e:
        print(f"[Warning] CNN prediction fallback: {e}")
        cnn_prob = 50.0

    # 2. OCR Text Extraction & Rule Checking
    extracted_text = extract_text_from_image(pil_img)
    rule_results = rule_based_verification(extracted_text)
    rule_score = rule_results['rule_score']

    # 3. Weighted Hybrid Fusion Formula: 60% CNN + 40% Rules
    final_risk_score = (0.60 * cnn_prob) + (0.40 * rule_score)

    is_fake = final_risk_score >= 50.0
    verdict = "FAKE" if is_fake else "GENUINE"
    confidence = final_risk_score if is_fake else (100.0 - final_risk_score)

    return {
        'verdict': verdict,
        'confidence': round(confidence, 1),
        'final_risk_score': round(final_risk_score, 1),
        'cnn_probability': round(cnn_prob, 1),
        'rule_risk_score': round(rule_score, 1),
        'red_flags': rule_results['red_flags'],
        'extracted_text': extracted_text[:300] + ('...' if len(extracted_text) > 300 else '')
    }

# ==============================================================================
# 3. GRADIO WEB INTERFACE
# ==============================================================================
def create_gradio_app(model):
    test_sample_paths = sorted(glob.glob(f"{TEST_DIR}/*.png"))

    def gradio_predict(input_image):
        if input_image is None:
            return "No image provided", "0%", "0 / 100", "Please upload a job advertisement poster.", ""

        res = verify_job_advertisement(input_image, model)

        verdict_badge = f"🚨 {res['verdict']} ADVERTISEMENT" if res['verdict'] == 'FAKE' else f"✅ {res['verdict']} ADVERTISEMENT"
        conf_str = f"{res['confidence']}%"
        risk_gauge = f"Overall Risk Score: {res['final_risk_score']} / 100  (CNN: {res['cnn_probability']}% | Rules: {res['rule_risk_score']}%)"

        red_flags_md = "### 🚩 Detected Red Flags & Explanations:\n"
        for i, flag in enumerate(res['red_flags'], 1):
            red_flags_md += f"**{i}.** {flag}\n\n"

        ocr_text = res['extracted_text'] or '[No text extracted]'
        ocr_preview = f"**Extracted OCR Text:**\n\n{ocr_text}"

        return verdict_badge, conf_str, risk_gauge, red_flags_md, ocr_preview

    with gr.Blocks(title="AI Fake Job Ad Detection & Verification System") as demo:
        gr.Markdown("""
        # 🛡️ AI Fake Job Advertisement Detection & Verification System
        ### Hybrid Multimodal Inspection: CNN Visual Classification (60%) + Pytesseract OCR NLP Verification (40%)
        *Upload any job ad poster or screenshot from your computer, or click a sample below to analyze risk factors.*
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="numpy", label="Upload Job Advertisement Image (Poster / Screenshot)")
                analyze_btn = gr.Button("🔍 Verify Advertisement Now", variant="primary", size="lg")

                if test_sample_paths:
                    gr.Examples(
                        examples=[[p] for p in test_sample_paths],
                        inputs=image_input,
                        label="Click a Pre-Generated Sample Poster to Test"
                    )

            with gr.Column(scale=1):
                verdict_output = gr.Textbox(label="Final Verification Verdict", text_align="center")
                confidence_output = gr.Textbox(label="System Confidence Level", text_align="center")
                risk_meter_output = gr.Textbox(label="Hybrid Risk Meter Breakdown (0-100)", text_align="center")
                red_flags_output = gr.Markdown()
                ocr_text_output = gr.Markdown()

        analyze_btn.click(
            fn=gradio_predict,
            inputs=[image_input],
            outputs=[verdict_output, confidence_output, risk_meter_output, red_flags_output, ocr_text_output]
        )

    return demo

if __name__ == '__main__':
    cnn_model = load_or_initialize_model()
    demo = create_gradio_app(cnn_model)
    demo.launch(share=False)
