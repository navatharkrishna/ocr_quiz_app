import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import cv2
import numpy as np
import openai
import pandas as pd
import json
import os
from github import Github

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="OCR Quiz Extractor", layout="wide")
st.title("📘 Marathi + English OCR Quiz Extractor & GitHub Uploader")

st.markdown("""
Upload a **PDF file** (Marathi or English), this app will:
1. Extract text using OCR (Tesseract)
2. Send text to GPT for cleaning & question formatting
3. Generate a CSV file in the desired format
4. Automatically push the CSV to your GitHub repo
""")

# -------------------------
# Fetch sensitive info from environment variables
# -------------------------
api_key = os.getenv("OPENAI_API_KEY")
github_token = os.getenv("GITHUB_TOKEN")
repo_name = os.getenv("GITHUB_REPO")
github_path = os.getenv("GITHUB_PATH")

if not all([api_key, github_token, repo_name, github_path]):
    st.error("❌ One or more environment variables are missing! Please set OPENAI_API_KEY, GITHUB_TOKEN, GITHUB_REPO, GITHUB_PATH.")
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
text_file_path = os.path.join(output_dir, "ocr_output.txt")
csv_file_path = os.path.join(output_dir, "quiz.csv")

# -------------------------
# Step 1: OCR PDF → Extract text
# -------------------------
st.info("🔍 Running OCR... Please wait.")
doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
output_text = ""

for page_num in range(len(doc)):
    page = doc[page_num]
    mat = fitz.Matrix(3, 3)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.dilate(gray, np.ones((1, 1), np.uint8), iterations=1)
    pil_img = Image.fromarray(gray)

    text = pytesseract.image_to_string(pil_img, lang="mar+eng")
    output_text += f"\n--- Page {page_num + 1} ---\n{text}\n"

with open(text_file_path, "w", encoding="utf-8") as f:
    f.write(output_text)

st.success("✅ OCR completed!")

# -------------------------
# Step 2: Send OCR text to GPT
# -------------------------
st.info("🤖 Formatting text using GPT...")

with open(text_file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

max_chunk_size = 3000
chunks = [raw_text[i:i + max_chunk_size] for i in range(0, len(raw_text), max_chunk_size)]
formatted_questions = []

for i, chunk in enumerate(chunks):
    st.write(f"Processing chunk {i+1} of {len(chunks)}...")

    prompt = """
You are given OCR-extracted quiz questions in Marathi and English.
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

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # or gpt-5-mini
        messages=[
            {"role": "system", "content": "You are a helpful assistant that structures quiz questions."},
            {"role": "user", "content": prompt + "\n\nText:\n" + chunk}
        ]
    )

    try:
        formatted_questions += json.loads(response.choices[0].message.content)
    except Exception as e:
        st.warning(f"⚠️ JSON parsing error in chunk {i+1}: {e}")

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

    st.success("✅ CSV generated!")
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
    g = Github(github_token)
    repo = g.get_repo(repo_name)

    try:
        contents = repo.get_contents(github_path)
        repo.update_file(contents.path, "Update quiz.csv via Streamlit OCR app", df.to_csv(index=False, encoding="utf-8-sig"), contents.sha, branch="main")
        st.success(f"✅ CSV updated at GitHub: {repo.html_url}/blob/main/{github_path}")
    except Exception:
        repo.create_file(github_path, "Create quiz.csv via Streamlit OCR app", df.to_csv(index=False, encoding="utf-8-sig"), branch="main")
        st.success(f"✅ CSV created at GitHub: {repo.html_url}/blob/main/{github_path}")
else:
    st.error("❌ No valid formatted data found. Check OCR output.")
