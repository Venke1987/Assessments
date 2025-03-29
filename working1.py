import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import openai
import json
from datetime import datetime
import fitz  # PyMuPDF for PDFs
import docx  # For Word document handling
import ast  # For checking Python syntax
import io  # For handling uploaded files
import sqlite3
import os
from io import BytesIO
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




# Load environment variables (if needed)
load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # Use Streamlit Secrets API Key


def init_db():
    conn = sqlite3.connect('student_quiz_history.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS quiz_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        student_name TEXT NOT NULL,
        topic TEXT NOT NULL,
        score INTEGER NOT NULL,
        total_questions INTEGER NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    return conn

def extract_text_from_docx(uploaded_file):
    try:
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def extract_text_from_pdf(uploaded_file):
    try:
        text = ""
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def extract_text_from_local_file(file_path):
    text = ""
    try:
        if file_path.lower().endswith(".pdf"):
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    text += page.get_text()
        elif file_path.lower().endswith(".docx"):
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        text = f"Error extracting text from {file_path}: {str(e)}"
    return text

def compute_local_plagiarism_scores(student_text, folder_path="local_reports"):
    similarities = {}
    if not os.path.exists(folder_path):
        return {"Error": f"Folder '{folder_path}' not found. Please create it and add your reports."}
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".pdf", ".docx")):
            file_path = os.path.join(folder_path, fname)
            local_text = extract_text_from_local_file(file_path)
            if local_text.startswith("Error extracting text"):
                similarities[fname] = local_text
            else:
                docs = [student_text, local_text]
                tfidf = TfidfVectorizer().fit_transform(docs)
                score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                similarities[fname] = score
    return similarities

init_db()

st.title("🚀 Generative AI-Based MEC102 Engineering Design Project Report Assessment System")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Dashboard",
    "🔍 Plagiarism/Reasoning Finder",
    "📈 Student Analytics"
])

if page == "🔍 Plagiarism/Reasoning Finder":
    st.header("🔍 Plagiarism/Reasoning Finder")
    uploaded_file = st.file_uploader("Upload student's document (.docx or .pdf)", type=["docx", "pdf"])

    student_submission = ""
    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1].lower()
        if file_type == "docx":
            student_submission = extract_text_from_docx(uploaded_file)
        elif file_type == "pdf":
            student_submission = extract_text_from_pdf(uploaded_file)

        st.subheader("📄 Extracted Submission:")
        st.text_area("Extracted Text", student_submission, height=300)

    rubric = {"Concept Understanding": 10, "Implementation": 10, "Analysis": 10, "Clarity": 5, "Creativity": 5}

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("📝 Generate AI Assessment"):
            if student_submission.strip():
                prompt = (
                    f"Assess the following submission based on this rubric: {json.dumps(rubric)}\n\n"
                    f"Submission:\n{student_submission}"
                )
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an AI grading student assignments."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                review = response.choices[0].message.content.strip()
                st.session_state.ai_assessment = review
                st.success("✅ AI Feedback Generated:")
                st.write(review)
            else:
                st.warning("⚠️ Please upload a valid document.")

    with col2:
        if st.button("🔍 Check for Plagiarism (LLM-based)"):
            if student_submission.strip():
                with st.spinner("Checking plagiarism..."):
                    plagiarism_prompt = (
                        f"Analyze the following document for plagiarism risk. "
                        f"Your response MUST include a clear plagiarism percentage in the first line with this exact format: "
                        f"'Plagiarism Risk: XX%' (where XX is a number between 0 and 100). "
                        f"Then provide your explanation below.\n\n"
                        f"Document:\n{student_submission}"
                    )
                    plagiarism_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an AI that checks for plagiarism. Always provide a clear plagiarism percentage in your first line."},
                            {"role": "user", "content": plagiarism_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=400
                    )
                    plagiarism_result = plagiarism_response.choices[0].message.content.strip()
                    st.session_state.llm_plagiarism = plagiarism_result

                    try:
                        if "Plagiarism Risk:" in plagiarism_result:
                            first_line = plagiarism_result.split('\n')[0]
                            percentage_text = first_line.split('Plagiarism Risk:')[1].strip()
                            if '%' in percentage_text:
                                percentage = float(percentage_text.replace('%', '').strip())
                                st.subheader("Plagiarism Risk Score:")
                                st.progress(percentage/100)
                                if percentage < 30:
                                    st.success(f"📊 Plagiarism Risk: {percentage}% (Low Risk)")
                                elif percentage < 60:
                                    st.warning(f"📊 Plagiarism Risk: {percentage}% (Medium Risk)")
                                else:
                                    st.error(f"📊 Plagiarism Risk: {percentage}% (High Risk)")
                    except Exception as e:
                        st.error(f"Error parsing plagiarism percentage: {str(e)}")
                    st.subheader("📝 Plagiarism Analysis:")
                    st.write(plagiarism_result)
            else:
                st.warning("⚠️ Please upload a valid document first.")

    with col3:
        if st.button("🔎 Compare with Local Reports"):
            if student_submission.strip():
                with st.spinner("Comparing text with local PDF/DOCX reports..."):
                    results = compute_local_plagiarism_scores(student_submission, "local_reports")
                st.session_state.local_similarity = results
                st.subheader("Local Similarity Scores (TF-IDF)")
                if "Error" in results and isinstance(results["Error"], str):
                    st.error(results["Error"])
                else:
                    items = sorted(
                        results.items(),
                        key=lambda x: x[1] if isinstance(x[1], float) else 0,
                        reverse=True
                    )
                    for fname, score in items:
                        if isinstance(score, float):
                            st.write(f"• **{fname}** → Similarity: {score:.2f}")
                            if score >= 0.8:
                                st.warning("High similarity! Potential plagiarism.")
                        else:
                            st.error(f"{fname}: {score}")
            else:
                st.warning("⚠️ Please upload a valid document first.")

    st.markdown("---")
    if st.button("📂 View All Scores"):
        st.subheader("📝 AI Assessment Feedback")
        if "ai_assessment" in st.session_state:
            st.info(st.session_state.ai_assessment)
        else:
            st.warning("No AI Assessment available.")

        st.subheader("📌 LLM-Based Plagiarism Result")
        if "llm_plagiarism" in st.session_state:
            st.info(st.session_state.llm_plagiarism)
        else:
            st.warning("No LLM-based plagiarism check performed.")

        st.subheader("📎 Local Report Similarity")
        if "local_similarity" in st.session_state:
            for fname, score in sorted(
                st.session_state.local_similarity.items(),
                key=lambda x: x[1] if isinstance(x[1], float) else 0,
                reverse=True
            ):
                if isinstance(score, float):
                    st.write(f"• **{fname}** → Similarity: {score:.2f}")
                    if score >= 0.8:
                        st.warning("⚠️ High similarity! Potential plagiarism.")
                else:
                    st.error(f"{fname}: {score}")
        else:
            st.warning("No local similarity check performed.")
elif page == "📈 Student Analytics":
    st.header("📈 Student Performance Analytics")
    
    student_id = st.selectbox("Select Student ID", list(students_data.keys()), key="analytics_select")
    student_name = students_data[student_id]["name"]
    st.subheader(f"Performance of: {student_name}")
    
    history_df = get_student_quiz_history(student_id)
    
    if not history_df.empty:
        st.markdown("### 🧪 Quiz Scores Over Time")
        chart_data = history_df[["timestamp", "score"]].copy()
        chart_data["timestamp"] = pd.to_datetime(chart_data["timestamp"])
        chart_data = chart_data.sort_values("timestamp")
        st.line_chart(chart_data.set_index("timestamp"))
        
        st.markdown("### 📝 All Quiz Attempts")
        st.dataframe(history_df)
    else:
        st.warning("No quiz history found for this student.")


