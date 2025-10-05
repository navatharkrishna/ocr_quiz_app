import streamlit as st
import fitz  # PyMuPDF
import openai
import pandas as pd
import json
from github import Github
import os

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

openai.api_key = api_key

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
# Step 1: Extract text from PDF (no Tesseract)
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

max_chunk_size = 3000
chunks = [output_text[i:i + max_chunk_size] for i in range(0, len(output_text), max_chunk_size)]
formatted_questions = []

for i, chunk in enumerate(chunks):
    st.write(f"Processing chunk {i+1} of {len(chunks)}...")
    prompt = """
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
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that structures quiz questions."},
                {"role": "user", "content": prompt + "\n\nText:\n" + chunk}
            ]
        )
        formatted_questions += json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        st.warning(f"⚠️ GPT/JSON error in chunk {i+1}: {e}")

# -------------------------
# Step 3: Convert JSON → CSV
# -------------------------
if formatted_questions:
    df = pd.DataFrame(formatted_questions)
    columns_order = ["question_no","question","option1","option2","option3","option4","correct_answer","description","reference"]
    for col in columns_order:
        if col not in df.columns:
            df[col] = ""
    df = df[columns_order]
    df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")

    st.success("✅ CSV generated!")
    st.dataframe(df.head())

    with open(csv_file_path, "rb") as f:
        st.download_button(label="📥 Download CSV", data=f, file_name="quiz.csv", mime="text/csv")

    # -------------------------
    # Step 4: Upload CSV to GitHub
    # -------------------------
    st.info("⬆️ Uploading CSV to GitHub...")
    g = Github(github_token)
    repo = g.get_repo(repo_name)

    csv_content = df.to_csv(index=False, encoding="utf-8-sig")
    try:
        contents = repo.get_contents(github_path)
        repo.update_file(
            path=contents.path,
            message="Update quiz.csv via Streamlit PDF app",
            content=csv_content,
            sha=contents.sha,
            branch="main"
        )
        st.success(f"✅ CSV updated at GitHub: {repo.html_url}/blob/main/{github_path}")
    except Exception:
        repo.create_file(
            path=github_path,
            message="Create quiz.csv via Streamlit PDF app",
            content=csv_content,
            branch="main"
        )
        st.success(f"✅ CSV created at GitHub: {repo.html_url}/blob/main/{github_path}")
else:
    st.error("❌ No valid formatted data found. Check extracted text.")
