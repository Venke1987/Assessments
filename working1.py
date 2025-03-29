
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import json
from datetime import datetime
import fitz  # PyMuPDF for PDFs
import docx  # For Word document handling
import io
import sqlite3
import os
from io import BytesIO
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import unicodedata

# Load environment variables
load_dotenv()

# OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Faculty Login ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def authenticate_user():
    st.sidebar.markdown("🔐 **Faculty Login**")
    password = st.sidebar.text_input("Enter Password", type="password")
    if password == "admin123":
        st.session_state.authenticated = True
        st.success("🔓 Access granted.")
    else:
        st.sidebar.warning("Invalid password.")

if not st.session_state.authenticated:
    authenticate_user()
    if not st.session_state.authenticated:
        st.stop()

# --- DB Functions ---
def init_db():
    conn = sqlite3.connect('student_quiz_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            student_name TEXT,
            topic TEXT,
            score INTEGER,
            total_questions INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

def get_student_quiz_history(student_id=None):
    conn = init_db()
    c = conn.cursor()
    if student_id:
        c.execute("SELECT * FROM quiz_results WHERE student_id=? ORDER BY timestamp DESC", (student_id,))
    else:
        c.execute("SELECT * FROM quiz_results ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=['id', 'student_id', 'student_name', 'topic', 'score', 'total_questions', 'timestamp'])

# --- Text Extraction Functions ---
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

def compute_local_plagiarism_scores(student_text, folder_path="local_reports"):
    similarities = {}
    if not os.path.exists(folder_path):
        return {"Error": "Folder not found."}
    for fname in os.listdir(folder_path):
        if fname.endswith((".pdf", ".docx")):
            try:
                path = os.path.join(folder_path, fname)
                with open(path, 'rb') as f:
                    if fname.endswith("pdf"):
                        text = extract_text_from_pdf(f)
                    else:
                        text = extract_text_from_docx(f)
                docs = [student_text, text]
                tfidf = TfidfVectorizer().fit_transform(docs)
                score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                similarities[fname] = round(score, 2)
            except:
                similarities[fname] = "Error"
    return similarities

def clean_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

# --- UI Section ---
st.title("🚀 Generative AI-Based MEC102 Engineering Design Report Assessment")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🔍 Plagiarism/Reasoning Finder", "📈 Student Analytics"])

if page == "🔍 Plagiarism/Reasoning Finder":
    st.header("📄 Report")
    uploaded_file = st.file_uploader("Upload student's report (.docx or .pdf)", type=["docx", "pdf"])
    student_id = st.text_input("Enter Student ID")
    ai_assessment = ""
    llm_plagiarism = ""
    local_similarities = {}

    if uploaded_file:
        ext = uploaded_file.name.split('.')[-1].lower()
        student_text = extract_text_from_pdf(uploaded_file) if ext == "pdf" else extract_text_from_docx(uploaded_file)

        rubric = st.text_area("✏️ Customize AI Feedback Rubric (JSON format)", value=json.dumps({
            "Concept Understanding": 10,
            "Implementation": 10,
            "Analysis": 10,
            "Clarity": 5,
            "Creativity": 5
        }, indent=2), height=150)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📝 Generate AI Assessment"):
                prompt = f"Evaluate based on rubric: {rubric}\n\nSubmission:\n{student_text}"
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an AI grading assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                ai_assessment = response.choices[0].message.content
                st.session_state['ai_assessment'] = ai_assessment
                st.success("✅ AI Feedback")
                st.write(ai_assessment)

        with col2:
            if st.button("🔍 LLM-Based Plagiarism"):
                prompt = f"Check for plagiarism and respond as: 'Plagiarism Risk: XX%\nDetails: ...'\n\nText:\n{student_text}"
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a plagiarism checker."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                llm_plagiarism = response.choices[0].message.content
                st.session_state['llm_plagiarism'] = llm_plagiarism
                st.info(llm_plagiarism)

        with col3:
            if st.button("🔎 Compare with Local Reports"):
                results = compute_local_plagiarism_scores(student_text)
                st.session_state['local_similarity'] = results
                for doc, score in results.items():
                    st.write(f"📄 {doc}: {score}")

    # PDF Export
    if st.button("📤 Export PDF Report"):
        if student_id:
            buffer = BytesIO()
            pdf = FPDF()
            pdf.add_page()

            # University logo
            if os.path.exists("sastra_logo.jpg"):
                pdf.image("sastra_logo.jpg", x=80, w=50)
                pdf.ln(10)

            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Engineering Design Assessment Report", ln=True, align='C')
            pdf.ln(10)

            pdf.set_font("Arial", '', 12)
            rubric_dict = json.loads(rubric)
            total_score = sum(rubric_dict.values())

            pdf.cell(0, 10, "Rubric-Based Evaluation:", ln=True)
            for k, v in rubric_dict.items():
                pdf.cell(0, 10, f"- {k}: {v}/10", ln=True)
            pdf.cell(0, 10, f"Total Score: {total_score}/40", ln=True)
            pdf.ln(10)

            ai_feedback = clean_text(st.session_state.get("ai_assessment", "N/A"))
            plagiarism_result = clean_text(st.session_state.get("llm_plagiarism", "N/A"))

            pdf.multi_cell(0, 10, f"AI Assessment:\n{ai_feedback}")
            pdf.ln(5)
            pdf.multi_cell(0, 10, f"LLM-Based Plagiarism Result:\n{plagiarism_result}")
            pdf.ln(5)
            pdf.multi_cell(0, 10, "Local Report Similarity")
            for fname, score in st.session_state.get("local_similarity", {}).items():
                pdf.cell(0, 10, f"{fname}: {score}", ln=True)

            pdf_output = buffer
            pdf.output(pdf_output, 'F')
            pdf_output.seek(0)
            st.download_button("📥 Download Styled Report", data=pdf_output, file_name=f"{student_id}_Assessment_Report.pdf")

        else:
            st.error("❗ Please enter Student ID before exporting the report.")

elif page == "📈 Student Analytics":
    st.header("📈 Student Performance Analytics")
    df = get_student_quiz_history()
    if not df.empty:
        student_ids = df["student_id"].unique()
        selected_student = st.selectbox("Select Student ID", student_ids)
        student_df = df[df["student_id"] == selected_student]
        student_df["timestamp"] = pd.to_datetime(student_df["timestamp"])
        student_df = student_df.sort_values("timestamp")

        st.subheader(f"Performance of {selected_student}")
        st.line_chart(student_df.set_index("timestamp")[["score"]])
        st.dataframe(student_df)
    else:
        st.info("No quiz history available.")

elif page == "📊 Dashboard":
    st.header("📊 Class Dashboard")
    df = get_student_quiz_history()
    if not df.empty:
        avg_score = df.groupby("student_id")["score"].mean().reset_index()
        st.bar_chart(avg_score.set_index("student_id"))
        st.dataframe(df)
    else:
        st.info("No quiz data found.")
