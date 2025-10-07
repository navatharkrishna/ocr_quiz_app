import streamlit as st
import fitz  # PyMuPDF
import openai
import httpx
import pandas as pd
import json
from github import Github
import os
import time

# -------------------------
# Streamlit App Config
# -------------------------
st.set_page_config(page_title="📘 Marathi + English OCR PDF Quiz Extractor", layout="wide")
st.title("📘 Marathi + English PDF Quiz Extractor & GitHub Uploader")

st.markdown("""
### 🧠 What this app does:
1️⃣ Extracts text from uploaded **PDF** (supports Marathi + English)  
2️⃣ Uses **GPT (gpt-4o-mini)** to format and structure quiz questions  
3️⃣ Converts to a clean **CSV file**  
4️⃣ Automatically uploads the result to your **GitHub repo 🚀**
""")

# -------------------------
# API & GitHub Credentials
# -------------------------
openai_api_key = st.secrets.get("OPENAI_API_KEY")
github_token = st.secrets.get("GITHUB_TOKEN")
github_repo = st.secrets.get("GITHUB_REPO", "navatharkrishna/ocr_quiz_app")

if not openai_api_key or not github_token:
    st.error("🚨 Missing API keys! Please add OPENAI_API_KEY and GITHUB_TOKEN in Streamlit secrets.")
    st.stop()

# -------------------------
# Fix for Proxy Issue
# -------------------------
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(proxy_var, None)

# Create OpenAI client safely
import httpx
from openai import OpenAI

def create_openai_client(api_key):
    transport = httpx.HTTPTransport(retries=3)
    client = OpenAI(api_key=api_key, http_client=httpx.Client(transport=transport, timeout=60.0))
    return client

client = create_openai_client(openai_api_key)

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("📄 Upload PDF file", type=["pdf"])
if not uploaded_file:
    st.info("Please upload a PDF file to start processing.")
    st.stop()

# -------------------------
# Step 1: Extract Text
# -------------------------
with st.spinner("🔍 Extracting text from PDF..."):
    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    extracted_text = ""
    for page in pdf_doc:
        extracted_text += page.get_text("text") + "\n"

if not extracted_text.strip():
    st.error("No readable text found in the PDF.")
    st.stop()

# Display a snippet
st.text_area("📜 Extracted Text Preview", extracted_text[:1500])

# -------------------------
# Step 2: Process via GPT
# -------------------------
st.subheader("✨ Processing Text with GPT...")

prompt = f"""
You are an AI that converts OCR quiz text into clean structured JSON.
The text may be in Marathi or English.

Output JSON format:
[
  {{
    "question_no": 1,
    "question": "Question text",
    "option1": "Option A",
    "option2": "Option B",
    "option3": "Option C",
    "option4": "Option D",
    "answer": "Correct option"
  }}
]

Text to process:
{extracted_text}
"""

with st.spinner("🧠 Thinking... extracting structured quiz data..."):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw_output = response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        st.stop()

# -------------------------
# Step 3: Parse JSON
# -------------------------
st.subheader("🧾 Parsing GPT Output")

try:
    json_start = raw_output.find("[")
    json_end = raw_output.rfind("]") + 1
    json_str = raw_output[json_start:json_end]
    quiz_data = json.loads(json_str)
except Exception:
    st.error("⚠️ Failed to parse JSON from GPT response.")
    st.text(raw_output)
    st.stop()

df = pd.DataFrame(quiz_data)
st.dataframe(df.head())

# -------------------------
# Step 4: Save CSV
# -------------------------
timestamp = time.strftime("%Y%m%d-%H%M%S")
csv_filename = f"quiz_{timestamp}.csv"
df.to_csv(csv_filename, index=False)
st.success(f"✅ CSV file created: {csv_filename}")

# -------------------------
# Step 5: Upload to GitHub
# -------------------------
st.subheader("🚀 Uploading to GitHub...")

try:
    g = Github(github_token)
    repo = g.get_repo(github_repo)
    with open(csv_filename, "rb") as f:
        content = f.read()
    repo.create_file(
        path=f"data/{csv_filename}",
        message=f"Add quiz file {csv_filename}",
        content=content,
        branch="master",
    )
    st.success(f"✅ File uploaded successfully to GitHub → `data/{csv_filename}`")
except Exception as e:
    st.error(f"GitHub Upload Failed: {e}")
