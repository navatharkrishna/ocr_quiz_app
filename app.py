import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
from openai import OpenAI
import pandas as pd
import json
import os
import requests
import base64
import time
import re
from PIL import Image

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="📘 Marathi + English OCR PDF Quiz Extractor", layout="wide")
st.title("📘 Marathi + English OCR PDF Quiz Extractor & GitHub Uploader")

st.markdown("""
Upload a **PDF file** (Marathi or English), this app will:

1. 🧠 Extract text using **OCR (Tesseract)**
2. 🤖 Use **GPT** to format questions properly
3. 📊 Generate a **CSV file**
4. 🚀 Automatically upload it to your **GitHub repository**
""")

# -------------------------
# Load secrets (no hardcoding)
# -------------------------
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    github_token = st.secrets["MY_GH_TOKEN"]
    repo_name = st.secrets["MY_GH_REPO"]
    github_path = st.secrets["MY_GH_PATH"]
except KeyError:
    st.error("❌ Missing secrets! Please add OPENAI_API_KEY, MY_GH_TOKEN, MY_GH_REPO, MY_GH_PATH in Streamlit secrets.")
    st.stop()

client = OpenAI()  # ✅ Works for OpenAI v1.51.0+

# -------------------------
# Upload PDF
# -------------------------
uploaded_pdf = st.file_uploader("📄 Upload PDF file", type=["pdf"])
if uploaded_pdf is None:
    st.stop()

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
text_file_path = os.path.join(output_dir, "ocr_text_output.txt")
csv_file_path = os.path.join(output_dir, "quiz.csv")

# -------------------------
# OCR Extraction
# -------------------------
st.info("🔍 Extracting text from PDF using OCR...")
try:
    images = convert_from_bytes(uploaded_pdf.read())
except Exception as e:
    st.error(f"❌ Failed to read PDF: {e}")
    st.stop()

ocr_text = ""
for i, img in enumerate(images):
    st.write(f"📄 Processing Page {i+1}...")
    text = pytesseract.image_to_string(img, lang="mar+eng")  # Marathi + English OCR
    ocr_text += f"\n--- Page {i+1} ---\n{text}\n"

with open(text_file_path, "w", encoding="utf-8") as f:
    f.write(ocr_text)
st.success("✅ OCR extraction completed!")

# -------------------------
# Split text into chunks
# -------------------------
max_chunk_size = 1500
lines = ocr_text.split("\n")
chunks, current_chunk = [], ""
for line in lines:
    if len(current_chunk) + len(line) + 1 > max_chunk_size:
        chunks.append(current_chunk)
        current_chunk = line + "\n"
    else:
        current_chunk += line + "\n"
if current_chunk.strip():
    chunks.append(current_chunk)

# -------------------------
# GPT Prompt Template
# -------------------------
prompt_template = """
You are given quiz questions extracted from a PDF in Marathi or English.
Return ONLY valid JSON — no explanation, no markdown.

Format:
[
  {
    "question_no": 1,
    "question": "Question text",
    "option1": "Option A",
    "option2": "Option B",
    "option3": "Option C",
    "option4": "Option D",
    "correct_answer": "Option number (1-4)",
    "description": "Explanation if available",
    "reference": "Source if available"
  }
]
"""

def clean_json_string(raw):
    text = raw.strip()
    match = re.search(r"(\[.*\])", text, re.DOTALL)
    if match:
        text = match.group(1)
    text = re.sub(r",\s*]", "]", text)
    text = re.sub(r",\s*}", "}", text)
    text = text.replace("“", "\"").replace("”", "\"")
    return text.strip()

# -------------------------
# Process each chunk with GPT
# -------------------------
formatted_questions = []
progress = st.progress(0)
status_text = st.empty()

for i, chunk in enumerate(chunks):
    status_text.text(f"🤖 Processing chunk {i+1}/{len(chunks)}...")
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a JSON-only quiz formatter."},
                    {"role": "user", "content": prompt_template + "\n\nText:\n" + chunk}
                ],
                temperature=0.2,
            )

            content = resp.choices[0].message.content.strip()
            json_clean = clean_json_string(content)

            try:
                data = json.loads(json_clean)
                if isinstance(data, list):
                    formatted_questions += data
                else:
                    st.warning(f"⚠️ Chunk {i+1} did not return a list.")
            except json.JSONDecodeError as e:
                st.warning(f"⚠️ Chunk {i+1} JSON error: {e}")
            break
        except Exception as e:
            st.warning(f"⚠️ GPT error (chunk {i+1}, attempt {attempt+1}): {e}")
            time.sleep(2 * (attempt + 1))

    progress.progress((i + 1) / len(chunks))

progress.empty()
status_text.text("✅ GPT formatting completed!")

# -------------------------
# Convert JSON → CSV
# -------------------------
if formatted_questions:
    df = pd.DataFrame(formatted_questions)
    required_cols = [
        "question_no", "question", "option1", "option2",
        "option3", "option4", "correct_answer", "description", "reference"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[required_cols]

    # ✅ Save without index
    df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")

    st.success("✅ CSV generated successfully!")
    st.dataframe(df.style.hide(axis='index'))

    with open(csv_file_path, "rb") as f:
        st.download_button("📥 Download CSV", f, file_name="quiz.csv", mime="text/csv")

    # -------------------------
    # Upload to GitHub
    # -------------------------
    st.info("⬆️ Uploading CSV to GitHub...")
    url = f"https://api.github.com/repos/{repo_name}/contents/{github_path}"
    headers = {"Authorization": f"token {github_token}"}
    content_bytes = base64.b64encode(open(csv_file_path, "rb").read()).decode("utf-8")

    # Check if file exists
    r = requests.get(url, headers=headers)
    data = {
        "message": "Upload quiz.csv via OCR Streamlit App",
        "content": content_bytes,
        "branch": "master"
    }

    if r.status_code == 200:
        sha = r.json().get("sha")
        data["sha"] = sha

    put_r = requests.put(url, headers=headers, json=data)
    if put_r.status_code in [200, 201]:
        st.success(f"✅ CSV uploaded to GitHub: https://github.com/{repo_name}/blob/master/{github_path}")
    else:
        st.error(f"❌ GitHub upload failed! {put_r.status_code}: {put_r.text}")
else:
    st.error("❌ No valid quiz data extracted. Check OCR text.")
