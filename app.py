import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import pytesseract
from pdf2image import convert_from_bytes
from github import Github
import io
import os
import time

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="📘 OCR Quiz Extractor", layout="wide")
st.title("📘 Marathi + English OCR PDF Quiz Extractor & GitHub Uploader")

st.markdown("""
### 🧠 What this app does:
1. Extracts text from PDF (Marathi or English) using OCR  
2. Uses GPT (gpt-4o-mini) to clean, structure, and format quiz data  
3. Generates a neat CSV file  
4. Automatically uploads it to your GitHub repo 🚀
""")

# -------------------------
# Load Secrets
# -------------------------
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    github_token = st.secrets["MY_GH_TOKEN"]
    repo_name = st.secrets["MY_GH_REPO"]
    github_path = st.secrets["MY_GH_PATH"]
except KeyError as e:
    st.error(f"❌ Missing secret: {e}. Please set all required secrets in Streamlit Cloud.")
    st.stop()

# -------------------------
# Safe OpenAI Initialization (Fixes Proxy Error)
# -------------------------
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    if proxy_var in os.environ:
        del os.environ[proxy_var]

client = OpenAI(api_key=api_key)

# -------------------------
# File Upload
# -------------------------
uploaded_pdf = st.file_uploader("📄 Upload your PDF file", type=["pdf"])
if uploaded_pdf is None:
    st.info("👆 Please upload a Marathi or English quiz PDF to begin.")
    st.stop()

st.info("🧩 Converting PDF pages to images...")

# Convert PDF → Images
try:
    images = convert_from_bytes(uploaded_pdf.read())
except Exception as e:
    st.error(f"❌ Error reading PDF: {e}")
    st.stop()

# -------------------------
# Step 1: OCR Extraction
# -------------------------
progress_bar = st.progress(0)
status_text = st.empty()

st.info("🔍 Extracting text from pages (OCR in progress...)")
full_text = ""

for i, img in enumerate(images):
    status_text.text(f"Processing page {i+1} of {len(images)}...")
    text = pytesseract.image_to_string(img, lang="mar+eng")
    full_text += f"\n--- Page {i+1} ---\n{text}"
    progress_bar.progress((i + 1) / len(images))

progress_bar.empty()
status_text.text("✅ OCR text extraction completed!")

# -------------------------
# Step 2: Process with GPT
# -------------------------
st.info("🤖 Formatting text using GPT to structured quiz format...")

prompt_template = """
You are an AI that converts raw OCR quiz text into structured JSON.

Output ONLY valid JSON array like this:
[
  {
    "question_no": 1,
    "question": "Which planet is known as the Red Planet?",
    "option1": "Earth",
    "option2": "Mars",
    "option3": "Venus",
    "option4": "Jupiter",
    "correct_answer": "Mars",
    "description": "Mars is known as the Red Planet due to iron oxide on its surface.",
    "reference": "Textbook Reference or N/A"
  }
]

Input text may be Marathi or English. Skip invalid or incomplete questions.
"""

chunks = [full_text[i:i+1500] for i in range(0, len(full_text), 1500)]
formatted_questions = []

for i, chunk in enumerate(chunks):
    st.write(f"🧠 Processing text chunk {i+1} of {len(chunks)}...")
    retry = 0
    while retry < 3:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": chunk}
                ],
                temperature=0
            )

            text_output = response.choices[0].message.content.strip()
            text_output = text_output.replace("```json", "").replace("```", "")

            st.text_area(f"🧾 GPT Output (Chunk {i+1})", text_output, height=150)
            data = json.loads(text_output)

            if isinstance(data, list):
                formatted_questions += data
                break
        except json.JSONDecodeError:
            st.warning("⚠️ JSON parse error — retrying...")
        except Exception as e:
            st.warning(f"⚠️ API Error: {e}")
        retry += 1
        time.sleep(2)

# -------------------------
# Step 3: Convert JSON → CSV
# -------------------------
if not formatted_questions:
    st.error("❌ No valid quiz data generated. Check OCR quality or prompt.")
    st.stop()

df = pd.DataFrame(formatted_questions)
columns = [
    "question_no", "question", "option1", "option2", "option3", "option4",
    "correct_answer", "description", "reference"
]
for col in columns:
    if col not in df.columns:
        df[col] = ""

df = df[columns]
csv_data = df.to_csv(index=False, encoding="utf-8-sig")

st.success("✅ Quiz CSV generated successfully!")
st.dataframe(df.head())

st.download_button("📥 Download CSV", data=csv_data, file_name="quiz.csv", mime="text/csv")

# -------------------------
# Step 4: Upload CSV to GitHub
# -------------------------
st.info("⬆️ Uploading CSV to your GitHub repo...")

try:
    g = Github(github_token)
    repo = g.get_repo(repo_name)
    try:
        contents = repo.get_contents(github_path)
        repo.update_file(
            contents.path,
            "Update quiz.csv via Streamlit OCR app",
            csv_data,
            contents.sha,
            branch="main"
        )
        st.success(f"✅ CSV updated at: https://github.com/{repo_name}/blob/main/{github_path}")
    except Exception:
        repo.create_file(
            github_path,
            "Create quiz.csv via Streamlit OCR app",
            csv_data,
            branch="main"
        )
        st.success(f"✅ CSV created at: https://github.com/{repo_name}/blob/main/{github_path}")
except Exception as e:
    st.error(f"❌ GitHub upload failed: {e}")
