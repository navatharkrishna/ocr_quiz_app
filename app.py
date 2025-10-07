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
1. Extract text directly from PDF (fast, no Tesseract needed)
2. Send text to GPT for cleaning & question formatting
3. Generate a CSV file in the desired format
4. Automatically push the CSV to your GitHub repo
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
    st.error("❌ Missing secrets! Set OPENAI_API_KEY, MY_GH_TOKEN, MY_GH_REPO, MY_GH_PATH in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# -------------------------
# File upload
# -------------------------
uploaded_pdf = st.file_uploader("📄 Upload PDF file", type=["pdf"])
if uploaded_pdf is None:
    st.stop()

# -------------------------
# Prepare output paths
# -------------------------
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
text_file_path = os.path.join(output_dir, "pdf_text_output.txt")
csv_file_path = os.path.join(output_dir, "quiz.csv")

# -------------------------
# Step 1: Extract text from PDF
# -------------------------
st.info("🔍 Extracting text from PDF... Please wait.")
try:
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
except Exception as e:
    st.error(f"❌ Failed to open PDF: {e}")
    st.stop()

output_text = ""
for page_num, page in enumerate(doc):
    text = page.get_text()
    if text.strip():
        output_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
    else:
        output_text += f"\n--- Page {page_num + 1} ---\n[No text detected, possibly an image]\n"

with open(text_file_path, "w", encoding="utf-8") as f:
    f.write(output_text)

st.success("✅ Text extraction completed!")

# -------------------------
# Step 2: Send extracted text to GPT
# -------------------------
st.info("🤖 Formatting text using GPT...")

# Split text into manageable chunks (avoid breaking sentences)
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

formatted_questions = []

prompt_template = """
You are given PDF-extracted quiz questions in Marathi and English.
Correct spelling/formatting and structure them in this JSON format:

[
  {
    "question_no": 1,
    "question": "Corrected question text",
    "option1": "Option A text",
    "option2": "Option B text",
    "option3": "Option C text",
    "option4": "Option D text",
    "correct_answer": "Correct answer or option number",
    "description": "Explanation if available",
    "reference": "Source / reference if available"
  }
]

Return only JSON array, no extra text.
"""

def clean_json_string(s):
    """Fix common JSON issues"""
    s = re.sub(r",\s*]", "]", s)
    s = re.sub(r",\s*}", "}", s)
    return s

progress = st.progress(0)
status_text = st.empty()

for i, chunk in enumerate(chunks):
    status_text.text(f"Processing chunk {i+1} of {len(chunks)}...")
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that structures quiz questions."},
                    {"role": "user", "content": prompt_template + "\n\nText:\n" + chunk}
                ],
                temperature=0.3,
            )

            content = response.choices[0].message.content.strip()
            if content:
                content_clean = clean_json_string(content)
                try:
                    data = json.loads(content_clean)
                    if isinstance(data, list):
                        formatted_questions += data
                    else:
                        st.warning(f"⚠️ Chunk {i+1} returned invalid JSON (not a list). Skipping.")
                except json.JSONDecodeError:
                    st.warning(f"⚠️ Chunk {i+1} JSON parsing error. Content:\n{content_clean[:500]}...")
            else:
                st.warning(f"⚠️ Chunk {i+1} returned empty response.")
            break
        except Exception as e:
            st.warning(f"⚠️ GPT API error in chunk {i+1}, attempt {attempt+1}: {e}")
            time.sleep(2 * (attempt + 1))
    progress.progress((i + 1) / len(chunks))

progress.empty()
status_text.text("✅ GPT formatting completed!")

# -------------------------
# Step 3: Convert JSON → CSV
# -------------------------
if formatted_questions:
    df = pd.DataFrame(formatted_questions)
    columns_order = [
        "question_no", "question", "option1", "option2",
        "option3", "option4", "correct_answer", "description", "reference"
    ]
    for col in columns_order:
        if col not in df.columns:
            df[col] = ""
    df = df[columns_order]
    df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")

    st.success("✅ CSV generated successfully!")
    st.dataframe(df.head())

    with open(csv_file_path, "rb") as f:
        st.download_button(
            label="📥 Download CSV",
            data=f,
            file_name="quiz.csv",
            mime="text/csv"
        )

    # -------------------------
    # Step 4: Upload CSV to GitHub
    # -------------------------
    st.info("⬆️ Uploading CSV to GitHub...")

    url = f"https://api.github.com/repos/{repo_name}/contents/{github_path}"
    headers = {"Authorization": f"token {github_token}"}
    content_bytes = base64.b64encode(open(csv_file_path, "rb").read()).decode("utf-8")

    r = requests.get(url, headers=headers)
    data = {
        "message": "Upload quiz.csv via Streamlit PDF app",
        "content": content_bytes,
        "branch": "main"
    }

    if r.status_code == 200:
        # File exists → update
        sha = r.json().get("sha")
        data["sha"] = sha
        put_r = requests.put(url, headers=headers, json=data)
    else:
        # Create new file
        put_r = requests.put(url, headers=headers, json=data)

    if put_r.status_code in [200, 201]:
        st.success(f"✅ CSV uploaded to GitHub: https://github.com/{repo_name}/blob/main/{github_path}")
    else:
        st.error(f"❌ GitHub upload failed! Status: {put_r.status_code} Response: {put_r.text}")

else:
    st.error("❌ No valid formatted data found. Check extracted text.")
