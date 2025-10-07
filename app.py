import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import json
import time
from github import Github
import os
import httpx
from openai import OpenAI

# -------------------------
# Streamlit App Config
# -------------------------
st.set_page_config(page_title="📘 Marathi + English OCR PDF Quiz Extractor", layout="wide")
st.title("📘 Marathi + English PDF Quiz Extractor & GitHub Uploader")

st.markdown("""
### 🧠 What this app does:
1️⃣ Extracts text from uploaded **PDF** (supports Marathi + English)  
2️⃣ Auto-corrects Marathi OCR text using GPT  
3️⃣ Converts to clean **quiz JSON + CSV**  
4️⃣ Uploads automatically to your **GitHub repo 🚀**
""")

# -------------------------
# Load Secrets
# -------------------------
openai_api_key = st.secrets.get("OPENAI_API_KEY")
github_token = st.secrets.get("MY_GH_TOKEN")
github_repo = st.secrets.get("MY_GH_REPO")
github_path = st.secrets.get("MY_GH_PATH", "data/quiz.csv")

if not openai_api_key or not github_token or not github_repo:
    st.error("🚨 Missing API keys or GitHub info in Streamlit Secrets!")
    st.stop()

# -------------------------
# Fix Proxy Issue (for Streamlit Cloud)
# -------------------------
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(proxy_var, None)

# -------------------------
# OpenAI Client (with retries)
# -------------------------
def create_openai_client(api_key):
    transport = httpx.HTTPTransport(retries=3)
    return OpenAI(api_key=api_key, http_client=httpx.Client(transport=transport, timeout=120.0))

client = create_openai_client(openai_api_key)

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("📄 Upload PDF file", type=["pdf"])
if not uploaded_file:
    st.info("Please upload a PDF file to start processing.")
    st.stop()

# -------------------------
# Step 1: Extract Text (Better accuracy)
# -------------------------
with st.spinner("🔍 Extracting full text from PDF..."):
    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    extracted_text = ""
    for page in pdf_doc:
        # Combine both text layers for better accuracy
        text = page.get_text("text", flags=1 | 2 | 8)  # Extract all available text
        extracted_text += text.strip() + "\n\n"

if not extracted_text.strip():
    st.error("❌ No readable text found in the uploaded PDF.")
    st.stop()

st.text_area("📜 Extracted Text Preview", extracted_text[:1500])

# -------------------------
# Step 2: Marathi Text Correction
# -------------------------
st.subheader("🪶 Correcting Marathi text using GPT...")

correction_prompt = f"""
You are a Marathi language expert.
Given this OCR extracted Marathi text, correct all grammar, spacing, and word recognition errors.
Do not translate it — just fix the Marathi text.

Text:
{extracted_text}
"""

with st.spinner("✨ Correcting Marathi text..."):
    try:
        correction_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": correction_prompt}],
            temperature=0.3,
        )
        corrected_text = correction_response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"⚠️ Marathi correction failed: {e}. Using raw text instead.")
        corrected_text = extracted_text

st.text_area("✅ Corrected Text Preview", corrected_text[:1500])

# -------------------------
# Step 3: Quiz JSON Extraction
# -------------------------
st.subheader("🧩 Extracting structured quiz data...")

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

prompt = f"{prompt_template}\n\nProcess this corrected text:\n{corrected_text}"

with st.spinner("🤖 Extracting quiz questions..."):
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
# Step 4: Parse JSON
# -------------------------
st.subheader("🧾 Parsing extracted quiz data...")

try:
    json_start = raw_output.find("[")
    json_end = raw_output.rfind("]") + 1
    json_str = raw_output[json_start:json_end]
    quiz_data = json.loads(json_str)
except Exception:
    st.error("⚠️ Failed to parse JSON output. Here’s what GPT returned:")
    st.text(raw_output)
    st.stop()

df = pd.DataFrame(quiz_data)

# Remove index column from display
st.dataframe(df, use_container_width=True, hide_index=True)

# -------------------------
# Step 5: Save as CSV (no index)
# -------------------------
timestamp = time.strftime("%Y%m%d-%H%M%S")
csv_filename = f"quiz_{timestamp}.csv"
df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
st.success(f"✅ CSV file created: {csv_filename}")

# -------------------------
# Step 6: Upload to GitHub
# -------------------------
st.subheader("🚀 Uploading file to GitHub...")

try:
    g = Github(github_token)
    repo = g.get_repo(github_repo)
    with open(csv_filename, "rb") as f:
        content = f.read()

    github_full_path = github_path.replace(".csv", f"_{timestamp}.csv")

    repo.create_file(
        path=github_full_path,
        message=f"Add quiz file {csv_filename}",
        content=content,
        branch="master",
    )

    st.success(f"✅ Successfully uploaded to GitHub → `{github_full_path}`")

except Exception as e:
    st.error(f"❌ GitHub Upload Failed: {e}")
