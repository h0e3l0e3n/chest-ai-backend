import sys
import os
import io

# Force UTF-8 encoding for Windows terminal to prevent translation crashes
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4' # Optimize PyTorch CPU boot latency
import re
from deep_translator import GoogleTranslator

# This tells the code where things are
DB_PATH = "./medical_kb"
PDF_FILE = "ERS_Handbook.pdf"

# This is the "Translator" that turns words into math the AI understands
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_library():
    if not os.path.exists(PDF_FILE):
        print("❌ I can't find the ERS_Handbook.pdf file in this folder!")
        return

    print("📖 Reading the book... please wait...")
    loader = PyPDFLoader(PDF_FILE)
    # We chop the book into small pieces so the AI doesn't get confused
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = loader.load_and_split(text_splitter)

    print("📦 Saving information to the database...")
    Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=DB_PATH)
    print("✅ Success! Your library is ready.")

# Map the Flutter dropdown strings to Google Translate codes
LANG_MAP = {
    'English': 'en',
    'Spanish (Español)': 'es',
    'French (Français)': 'fr',
    'Mandarin (普通话)': 'zh-CN',
    'Arabic (العربية)': 'ar',
    'Hindi (हिन्दी)': 'hi'
}

def ask_library(question, role="General User", language="English"):
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    results = db.similarity_search(question, k=2)
    
    target_lang = LANG_MAP.get(language, 'en')
    
    # Translate the condition name if not English
    display_question = question
    if target_lang != 'en':
        try:
            display_question = GoogleTranslator(source='en', target=target_lang).translate(question)
        except:
            pass

    if role == "Medical Professional":
        header = f"CLINICAL ASSESSMENT OVERVIEW ({language})\n\n"
        header += f"Based on standard respiratory guidelines, the following clinical findings are relevant to {display_question}:\n\n"
    else:
        header = f"SIMPLE HEALTH OVERVIEW ({language})\n\n"
        header += f"Here is an easy-to-understand summary of {display_question} based on medical literature:\n\n"

    raw_explanation = ""
    # Synthesize and Clean the chunks professionally
    for res in results:
        text = res.page_content
        text = re.sub(r'', '', text) # Remove invalid encodings
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
        if clean_sentences:
            raw_explanation += " ".join(clean_sentences) + " "

    # Fallback if empty
    if not raw_explanation.strip():
        if role == "Medical Professional":
            raw_explanation = f"General review of the ERS Handbook literature suggests {question} involves recognized respiratory alteration. Clinical correlation and advanced imaging checkup are standard procedures."
        else:
            raw_explanation = f"According to the medical texts, {question} relates to changes in lung function. It is highly recommended to have a physician review this finding to ensure your safety."

    # Translate the entire explanation block
    final_explanation = raw_explanation
    if target_lang != 'en':
        try:
            # Deep translator handles roughly 5000 chars per call, which is plenty for our chunks
            final_explanation = GoogleTranslator(source='en', target=target_lang).translate(raw_explanation)
        except Exception as e:
            print(f"Translation Error: {e}", file=sys.stderr)

    full_output = header + final_explanation
    
    if role != "Medical Professional":
        disclaimer = "\n\nDisclaimer: This is an AI-generated summary. Always consult your doctor for real medical advice."
        if target_lang != 'en':
            try:
                disclaimer = "\n\n" + GoogleTranslator(source='en', target=target_lang).translate("Disclaimer: This is an AI-generated summary. Always consult your doctor for real medical advice.")
            except:
                pass
        final_explanation += disclaimer
        
    print(f"@@DIAG@@{display_question}")
    print(f"@@EXPL@@{header}{final_explanation.strip()}")

if __name__ == "__main__":
    if len(sys.argv) > 3:
        ask_library(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1:
        ask_library(sys.argv[1])
    else:
        build_library()