
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
import ast
import io
import sqlite3
import os
from io import BytesIO
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import unicodedata
import re

# Load environment variables
load_dotenv()

openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def authenticate_user():
    st.sidebar.markdown("🔐 **Faculty Login**")
    password = st.sidebar.text_input("Enter Password", type="password")
    if password == "admin@123":
        st.session_state.authenticated = True
        st.success("🔓 Access granted.")
    else:
        st.sidebar.warning("Invalid password.")

if not st.session_state.authenticated:
    authenticate_user()
    if not st.session_state.authenticated:
        st.stop()

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

st.title("🚀 Generative AI-Based MEC102 Engineering Design Report Assessment")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🔍 Plagiarism/Reasoning Finder", "📈 Student Analytics"])

def clean_text(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

if page == "🔍 Plagiarism/Reasoning Finder":
    st.header("📄 Upload and Assess Report")
    uploaded_file = st.file_uploader("Upload student's report (.docx or .pdf)", type=["docx", "pdf"])
    student_id = st.text_input("Enter Student ID", key="student_id_input", value="SEEE001")
    ai_assessment = ""
    llm_plagiarism = ""
    local_similarities = {}

    if uploaded_file:
        ext = uploaded_file.name.split('.')[-1].lower()
        student_text = extract_text_from_pdf(uploaded_file) if ext == "pdf" else extract_text_from_docx(uploaded_file)

        rubric_json = json.dumps({
            "Concept Understanding": 10,
            "Implementation": 10,
            "Analysis": 10,
            "Clarity": 5,
            "Creativity": 5
        }, indent=2)

        st.session_state["rubric_json"] = rubric_json
        rubric = st.text_area("✏️ Customize AI Feedback Rubric (JSON format)", value=rubric_json, height=150)

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
                prompt = f"Check for plagiarism and respond as: 'Plagiarism Risk: XX%'\n\nText:\n{student_text}"
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

    if st.button("📤 Export PDF Report"):
        ai_feedback = clean_text(st.session_state.get("ai_assessment", "Not available"))
        plagiarism_result = clean_text(st.session_state.get("llm_plagiarism", "Not available"))
        rubric_dict = json.loads(st.session_state.get("rubric_json", json.dumps({
            "Concept Understanding": 10,
            "Implementation": 10,
            "Analysis": 10,
            "Clarity": 5,
            "Creativity": 5
        })))
        local_similarity = st.session_state.get("local_similarity", {})
        total_score = sum(rubric_dict.values())

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        logo_path = "sastra_logo.jpg"
        if os.path.exists(logo_path):
            pdf.image(logo_path, x=45, y=10, w=125)
            pdf.ln(30)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Assessment Report for Student ID: {student_id}", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, "Rubric-Based Evaluation:", ln=True)
        for k, v in rubric_dict.items():
            pdf.cell(0, 10, f"- {k}: {v}/10", ln=True)
        pdf.cell(0, 10, f"Total Score: {total_score}/40", ln=True)
        pdf.ln(10)

        pdf.multi_cell(0, 10, f"AI Assessment:\n{ai_feedback}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"LLM-Based Plagiarism Result:\n{plagiarism_result}")
        pdf.ln(5)

        pdf.cell(0, 10, "Local Report Similarity:", ln=True)
        for fname, score in local_similarity.items():
            pdf.cell(0, 10, f"{fname}: {score}", ln=True)

        pdf_output = BytesIO()
        pdf_bytes = pdf.output(dest="S").encode("latin1")
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)

        st.download_button(
            label="📥 Download Styled Report",
            data=pdf_output,
            file_name=f"{student_id}_Assessment_Report.pdf",
            mime="application/pdf"
        )
        # Save assessment to quiz history
        conn = init_db()
        c = conn.cursor()
        c.execute("""
            INSERT INTO quiz_results (student_id, student_name, topic, score, total_questions)
            VALUES (?, ?, ?, ?, ?)
        """, (
            student_id,
            student_id,  # Replace with actual name if available
            "MEC102 Report",
            total_score,
            50  # or sum of max rubric values
        ))
        conn.commit()
        conn.close()

elif page == "📈 Student Analytics":
    st.header("📈 Student Performance Analytics")

    df = get_student_quiz_history()
    if df.empty:
        st.warning("No quiz history available.")
    else:
        student_ids = df['student_id'].unique().tolist()
        selected_id = st.selectbox("Select Student ID", student_ids)

        student_df = df[df['student_id'] == selected_id]
        student_name = student_df['student_name'].iloc[0]
        st.subheader(f"Performance Analytics for: {student_name} ({selected_id})")

        # Trend chart
        student_df["timestamp"] = pd.to_datetime(student_df["timestamp"])
        student_df = student_df.sort_values("timestamp")
        st.line_chart(student_df.set_index("timestamp")[["score"]])

        # Table
        st.dataframe(student_df)

        # Summary stats
        st.markdown("### Summary")
        st.write(f"**Total Quizzes Attempted**: {len(student_df)}")
        st.write(f"**Average Score**: {student_df['score'].mean():.2f}")
        st.write(f"**Best Score**: {student_df['score'].max()}")
        st.write(f"**Most Recent Topic**: {student_df['topic'].iloc[-1]}")

elif page == "📊 Dashboard":
    st.header("📊 Class-Wide Dashboard")
    df = get_student_quiz_history()

    if df.empty:
        st.info("No quiz data available.")
    else:
        st.subheader("Average Scores per Student")
        avg_score = df.groupby("student_id")["score"].mean().reset_index()
        st.bar_chart(avg_score.set_index("student_id"))

        st.subheader("Quiz Topic Distribution")
        topic_counts = df["topic"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=topic_counts.values, y=topic_counts.index, ax=ax)
        ax.set_xlabel("Number of Quizzes")
        ax.set_ylabel("Topic")
        st.pyplot(fig)

        st.subheader("Top Performing Students")
        top_students = avg_score.sort_values("score", ascending=False).head(5)
        st.table(top_students.rename(columns={"score": "Average Score"}))
# 🧪 TEMPORARY: Add dummy quiz history for testing
if "seeded" not in st.session_state:
    st.session_state.seeded = True
    conn = init_db()
    c = conn.cursor()
    sample_data = [
        ("SEEE001", "Adhithya V", "Edge AI", 8, 10),
        ("SEEE001", "Adhithya V", "Fuzzy Logic", 7, 10),
        ("SEEE010", "Dodda Sri Pujitha", "Edge AI", 9, 10),
        ("SEEE010", "Dodda Sri Pujitha", "Fuzzy Logic", 8, 10),
        ("SEEE011", "Gowtham", "AI Ethics", 6, 10),
    ]
    for student_id, student_name, topic, score, total_q in sample_data:
        c.execute("INSERT INTO quiz_results (student_id, student_name, topic, score, total_questions) VALUES (?, ?, ?, ?, ?)",
                  (student_id, student_name, topic, score, total_q))
    conn.commit()
    conn.close()
    st.success("Dummy data seeded! Refresh the app to view analytics.")
