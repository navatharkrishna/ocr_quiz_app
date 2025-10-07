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
2️⃣ Corrects Marathi sentences automatically using **GPT**  
3️⃣ Converts OCR text to structured quiz questions (JSON → CSV)  
4️⃣ Uploads CSV to your **GitHub repo 🚀**
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
# Safe OpenAI Initialization
# -------------------------
def create_openai_client(api_key):
    transport = httpx.HTTPTransport(retries=3)
    return OpenAI(api_key=api_key, http_client=httpx.Client(transport=transport, timeout=60.0))

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
    st.error("❌ No readable text found in the uploaded PDF.")
    st.stop()

st.text_area("📜 Extracted Text Preview", extracted_text[:1500])

# -------------------------
# Step 2: Marathi Text Correction (Batch-wise)
# -------------------------
st.subheader("🪶 Correcting Marathi text using GPT...")

def correct_marathi_text_in_batches(text, batch_size=2500):
    paragraphs = text.split("\n\n")
    chunks, current, count = [], "", 0
    for p in paragraphs:
        current += p + "\n\n"
        count += len(p)
        if count > batch_size:
            chunks.append(current.strip())
            current, count = "", 0
    if current:
        chunks.append(current.strip())

    corrected_chunks = []
    for i, chunk in enumerate(chunks, start=1):
        st.write(f"🔹 Processing batch {i}/{len(chunks)}...")
        prompt = f"""
        You are a Marathi language expert.
        Correct grammar, spacing, and OCR errors in the following Marathi text.
        Do NOT translate; keep it in Marathi.

        Text:
        {chunk}
        """

        for attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                corrected = resp.choices[0].message.content.strip()
                corrected_chunks.append(corrected)
                break
            except Exception as e:
                if attempt == 1:
                    st.warning(f"⚠️ Skipped batch {i} due to: {e}")
                time.sleep(2)

    return "\n\n".join(corrected_chunks)

with st.spinner("✨ Correcting Marathi text in batches..."):
    try:
        corrected_text = correct_marathi_text_in_batches(extracted_text)
    except Exception as e:
        st.warning(f"⚠️ Marathi correction failed: {e}. Using raw text instead.")
        corrected_text = extracted_text

st.text_area("✅ Corrected Text Preview", corrected_text[:1500])

# -------------------------
# Step 3: GPT Prompt Template for Quiz
# -------------------------
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

# Combine with corrected text
prompt = f"{prompt_template}\n\nProcess this input text:\n{corrected_text}"

# -------------------------
# Step 4: Convert to structured quiz using GPT
# -------------------------
st.subheader("✨ Converting text to structured quiz format...")

with st.spinner("🤖 Processing with GPT..."):
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
# Step 5: Parse JSON
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

# Remove index column and ensure correct order
columns = ["question_no", "question", "option1", "option2", "option3", "option4",
           "correct_answer", "description", "reference"]
for col in columns:
    if col not in df.columns:
        df[col] = ""
df = df[columns]

st.dataframe(df.head())

# -------------------------
# Step 6: Save as CSV
# -------------------------
timestamp = time.strftime("%Y%m%d-%H%M%S")
csv_filename = f"quiz_{timestamp}.csv"
df.to_csv(csv_filename, index=False)
st.success(f"✅ CSV file created: {csv_filename}")

# -------------------------
# Step 7: Upload to GitHub
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
