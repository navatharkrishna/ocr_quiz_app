import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
import pandas as pd
import json
import os
import requests
import base64
import time
import re

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="OCR Quiz Extractor", layout="wide")
st.title("📘 Marathi + English PDF Quiz Extractor & GitHub Uploader")

st.markdown("""
Upload a **PDF file** (Marathi or English), this app will:
1. Extract text directly from PDF  
2. Send text to GPT for cleaning & question formatting  
3. Generate a CSV file  
4. Automatically push it to your GitHub repo  
""")

# -------------------------
# Secrets
# -------------------------
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    github_token = st.secrets["MY_GH_TOKEN"]
    repo_name = st.secrets["MY_GH_REPO"]
    github_path = st.secrets["MY_GH_PATH"]
except KeyError:
    st.error("❌ Missing secrets! Please add OPENAI_API_KEY, MY_GH_TOKEN, MY_GH_REPO, MY_GH_PATH in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# -------------------------
# Upload PDF
# -------------------------
uploaded_pdf = st.file_uploader("📄 Upload PDF file", type=["pdf"])
if uploaded_pdf is None:
    st.stop()

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
text_file_path = os.path.join(output_dir, "pdf_text_output.txt")
csv_file_path = os.path.join(output_dir, "quiz.csv")

# -------------------------
# Extract text from PDF
# -------------------------
st.info("🔍 Extracting text from PDF...")
try:
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
except Exception as e:
    st.error(f"❌ Failed to open PDF: {e}")
    st.stop()

output_text = ""
for i, page in enumerate(doc):
    text = page.get_text()
    output_text += f"\n--- Page {i+1} ---\n{text if text.strip() else '[No text detected]'}\n"

with open(text_file_path, "w", encoding="utf-8") as f:
    f.write(output_text)
st.success("✅ Text extraction completed!")

# -------------------------
# Split into manageable chunks
# -------------------------
max_chunk_size = 1500
lines = output_text.split("\n")
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
You are given PDF-extracted quiz questions in Marathi and English.
Return ONLY valid JSON, no explanation or extra text.

JSON format:
[
  {
    "question_no": 1,
    "question": "Corrected question text",
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
    """Cleans GPT responses into valid JSON arrays"""
    text = raw.strip()

    # Extract only the JSON portion if GPT adds extra commentary
    match = re.search(r"(\[.*\])", text, re.DOTALL)
    if match:
        text = match.group(1)

    # Fix common JSON syntax issues
    text = re.sub(r",\s*]", "]", text)
    text = re.sub(r",\s*}", "}", text)
    text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    text = text.strip()

    return text

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
                    {"role": "system", "content": "You are a precise JSON-only formatter."},
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
                    st.warning(f"⚠️ Chunk {i+1} not a list. Skipped.")
            except json.JSONDecodeError as e:
                st.warning(f"⚠️ Chunk {i+1} JSON parse error: {e}\nPreview:\n{json_clean[:400]}")

            break  # stop retry loop
        except Exception as e:
            st.warning(f"⚠️ GPT error (chunk {i+1}, try {attempt+1}): {e}")
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
    df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")

    st.success("✅ CSV generated successfully!")
    st.dataframe(df.head())

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
        "message": "Upload quiz.csv via Streamlit PDF app",
        "content": content_bytes,
        "branch": "master"  # ✅ Use master branch if your repo default is master
    }

    if r.status_code == 200:
        sha = r.json().get("sha")
        data["sha"] = sha  # update existing file
    put_r = requests.put(url, headers=headers, json=data)

    if put_r.status_code in [200, 201]:
        st.success(f"✅ CSV uploaded to GitHub: https://github.com/{repo_name}/blob/master/{github_path}")
    else:
        st.error(f"❌ GitHub upload failed! {put_r.status_code}: {put_r.text}")
else:
    st.error("❌ No valid data generated. Check extracted text.")
