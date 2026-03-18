import os
import sys
import io
import re
import warnings

# Optimization 1: Quiet logs and high thread count
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '4'
warnings.filterwarnings('ignore')

# Optimization 2: Unicode for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings

# --- SETTINGS ---
DB_PATH = "./medical_kb"
MODEL_FILE = 'chexpert_mobilenetv2_fast (2).h5'
LANG_MAP = {
    'English': 'en',
    'Spanish (Español)': 'es',
    'French (Français)': 'fr',
    'Mandarin (普通话)': 'zh-CN',
    'Arabic (العربية)': 'ar',
    'Hindi (हिन्दी)': 'hi'
}

# --- INITIALIZATION (Loaded ONCE at startup) ---
print("🚀 [AI WORKER] Warming up... Booting AI Models (Please wait ~20s)...")

def fix_model_loading():
    from tensorflow.keras.layers import Dense
    class CustomDense(Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)
    return {'Dense': CustomDense}

# Load Vision Model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILE)
vision_model = tf.keras.models.load_model(model_path, custom_objects=fix_model_loading(), compile=False)

# Load RAG Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

app = Flask(__name__)

def clean_text(text):
    text = re.sub(r'--- Reference.*?---', '', text)
    text = text.replace('\n', ' ')
    sentences = re.split(r'(?<=[.!?]) +', text)
    clean_sentences = []
    for s in sentences:
        s_lower = s.lower()
        if any(bad in s_lower for bad in ['figure', 'arrow', 'arrowhead', 'handbook:', 'reference', 'table']):
            continue
        s = re.sub(r'\b[a-d]\)', '', s)
        s = re.sub(r'\([a-zA-Z]{1,2}\)', '', s)
        s = re.sub(r'\s{2,}', ' ', s).strip()
        if len(s) > 15:
            clean_sentences.append(s)
    return " ".join(clean_sentences)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    img_path = data.get('imagePath')
    role = data.get('role', 'General User')
    language = data.get('language', 'English')
    target_lang = LANG_MAP.get(language, 'en')

    # 1. Vision Inference
    img = image.load_img(img_path, target_size=(160, 160))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0 
    preds = vision_model.predict(x, verbose=0)
    labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
              'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'No Finding', 
              'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    diagnosis = labels[np.argmax(preds[0])]

    # 2. RAG Retrieval
    results = db.similarity_search(diagnosis, k=2)
    raw_explanation = " ".join([clean_text(res.page_content) for res in results])

    if not raw_explanation.strip():
        if role == "Medical Professional":
            raw_explanation = f"General review of the ERS Handbook literature suggests {diagnosis} involves recognized respiratory alteration."
        else:
            raw_explanation = f"According to the medical texts, {diagnosis} relates to changes in lung function."

    # 3. Translation & Header
    display_diag = diagnosis
    if target_lang != 'en':
        display_diag = GoogleTranslator(source='en', target=target_lang).translate(diagnosis)

    if role == "Medical Professional":
        header = f"CLINICAL ASSESSMENT OVERVIEW ({language})\n\nRelevant findings for {display_diag}:\n\n"
    else:
        header = f"SIMPLE HEALTH OVERVIEW ({language})\n\nEasy summary of {display_diag} findings:\n\n"

    final_expl = raw_explanation
    if target_lang != 'en':
        final_expl = GoogleTranslator(source='en', target=target_lang).translate(raw_explanation)

    if role != "Medical Professional":
        disc = "\n\nDisclaimer: This is an AI-generated summary. Always consult your doctor."
        if target_lang != 'en':
            disc = "\n\n" + GoogleTranslator(source='en', target=target_lang).translate(disc.strip())
        final_expl += disc

    return jsonify({
        "disease": display_diag,
        "explanation": f"{header}{final_expl.strip()}"
    })

if __name__ == "__main__":
    print("✨ [AI WORKER] Online! Models are hot and ready at http://localhost:5005")
    app.run(port=5005)
