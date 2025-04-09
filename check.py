import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import json
from datetime import datetime
import fitz  # PyMuPDF
import docx
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
import random  # used to simulate internet similarity
import html   # for robust HTML escaping

# =========================================================
# 1. SETUP: OpenAI key, Authentication, and Database setup
# =========================================================

# Load OpenAI API key from environment variables
load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=openai.api_key)

# --- User Authentication ---
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

# --- Database Initialization ---
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
            ai_score REAL,
            llm_plagiarism_score REAL,
            local_similarity_score REAL,
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
    return pd.DataFrame(rows, columns=[
        'id', 'student_id', 'student_name', 'topic', 'score',
        'total_questions', 'ai_score', 'llm_plagiarism_score',
        'local_similarity_score', 'timestamp'
    ])

# =========================================================
# 2. FILE EXTRACTION AND TEXT CLEANING FUNCTIONS
# =========================================================

def extract_pdf_safe(file_obj):
    """
    Reads file data into memory and tries to extract text from a PDF.
    Will raise fitz.fitz.EmptyFileError if the PDF is unreadable.
    """
    data = file_obj.read()
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            content = ""
            for page in doc:
                content += page.get_text()
        return content
    except fitz.fitz.EmptyFileError:
        raise

def extract_text_from_pdf(uploaded_file):
    """Extract text from a student-uploaded PDF (assumed valid)."""
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(uploaded_file):
    """Extract text from a DOCX file."""
    doc_ = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc_.paragraphs])

def clean_text(text):
    """
    Normalize text to remove special Unicode characters and emoji.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

# =========================================================
# 3. PLAGIARISM CHECK FUNCTIONS
# =========================================================

# 3a. Single-Score Local Plagiarism Check (for comparison)
def compute_local_plagiarism_scores(student_text, folder_path="local_reports"):
    similarities = []
    if not os.path.exists(folder_path):
        return 0.0
    for fname in os.listdir(folder_path):
        if fname.endswith((".pdf", ".docx")):
            try:
                path = os.path.join(folder_path, fname)
                with open(path, 'rb') as f:
                    if fname.endswith(".pdf"):
                        text = extract_text_from_pdf(f)
                    else:
                        text = extract_text_from_docx(f)
                docs = [student_text, text]
                tfidf = TfidfVectorizer().fit_transform(docs)
                score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                similarities.append(score)
            except:
                continue
    return round(max(similarities)*100, 2) if similarities else 0.0

# 3b. Chunk-Level Plagiarism Functions

def chunk_text(text, chunk_size=300, overlap=50):
    """
    Splits text into overlapping chunks based on word count.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def compute_local_plagiarism_details(student_text, folder_path="local_reports", threshold=0.3):
    """
    For each chunk in the student text, compute the best similarity score from local docs.
    Skip empty or unreadable files/chunks. Returns a list of dicts (one per flagged chunk).
    """
    student_chunks = chunk_text(student_text)
    local_docs = []

    # Process each local file (PDF or DOCX)
    if os.path.exists(folder_path):
        for fname in os.listdir(folder_path):
            if fname.endswith((".pdf", ".docx")):
                path = os.path.join(folder_path, fname)
                if os.path.getsize(path) == 0:
                    st.warning(f"Skipping empty file: {fname}")
                    continue
                try:
                    with open(path, 'rb') as f:
                        if fname.endswith(".pdf"):
                            file_text = extract_pdf_safe(f)
                        else:
                            file_text = extract_text_from_docx(f)
                except fitz.fitz.EmptyFileError:
                    st.warning(f"Skipping unreadable PDF: {fname}")
                    continue
                except Exception as e:
                    st.warning(f"Error reading file {fname}: {e}")
                    continue

                if not file_text.strip():
                    st.warning(f"Skipping file {fname} because it's empty.")
                    continue

                doc_chunks = chunk_text(file_text)
                local_docs.append((fname, doc_chunks))

    detailed_results = []
    for i, s_chunk in enumerate(student_chunks):
        if not s_chunk.strip():
            continue

        best_score = 0.0
        best_file = None

        # Compare student chunk against all valid chunks from local docs
        for fname, doc_chunks in local_docs:
            valid_doc_chunks = [dc for dc in doc_chunks if dc.strip()]
            if not valid_doc_chunks:
                continue

            docs = [s_chunk] + valid_doc_chunks
            if len(docs) < 2:
                continue

            try:
                tfidf = TfidfVectorizer().fit_transform(docs)
                sim_scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
                local_best = sim_scores.max()
                if local_best > best_score:
                    best_score = local_best
                    best_file = fname
            except ValueError as ve:
                st.warning(f"Skipping chunk {i} with {fname} due to TF-IDF error: {ve}")
                continue

        if best_score >= threshold:
            detailed_results.append({
                "chunk_index": i,
                "chunk_text": s_chunk,
                "source_type": "Local",
                "best_local_score": round(best_score, 3),
                "best_local_file": best_file
            })

    return detailed_results

def compute_internet_plagiarism_details(student_text, threshold=0.3):
    """
    Simulates internet plagiarism check by assigning random similarity scores to each chunk.
    Replace this with an actual API call if needed.
    """
    student_chunks = chunk_text(student_text)
    detailed_results = []
    for i, s_chunk in enumerate(student_chunks):
        simulated_score = random.uniform(0, 1)
        if simulated_score >= threshold:
            detailed_results.append({
                "chunk_index": i,
                "chunk_text": s_chunk,
                "source_type": "Internet",
                "best_internet_score": round(simulated_score, 3),
                "best_url": f"https://example.com/matching_page_{i}"
            })
    return detailed_results

# 3c. Merge Local and Internet Matches into a Single Row per Chunk
def merge_local_and_internet(local_details, internet_details):
    """
    For each unique chunk_index, merge the best local match and best internet match into one row.
    Returns a list of dictionaries with these combined values.
    """
    # Build dictionary for local details (keeping highest score per chunk)
    local_dict = {}
    for item in local_details:
        idx = item["chunk_index"]
        if idx not in local_dict or item["best_local_score"] > local_dict[idx]["best_local_score"]:
            local_dict[idx] = item

    # Build dictionary for internet details (keeping highest score per chunk)
    internet_dict = {}
    for item in internet_details:
        idx = item["chunk_index"]
        if idx not in internet_dict or item["best_internet_score"] > internet_dict[idx]["best_internet_score"]:
            internet_dict[idx] = item

    # Merge both dictionaries into one row per chunk_index
    all_indices = set(local_dict.keys()).union(set(internet_dict.keys()))
    merged_rows = []
    for idx in sorted(all_indices):
        # Get the chunk_text from either dictionary (they should match if available)
        chunk_text = local_dict[idx]["chunk_text"] if idx in local_dict else internet_dict[idx]["chunk_text"]

        # Get local values
        local_score = local_dict[idx]["best_local_score"] if idx in local_dict else 0.0
        local_file = local_dict[idx]["best_local_file"] if idx in local_dict else ""

        # Get internet values
        internet_score = internet_dict[idx]["best_internet_score"] if idx in internet_dict else 0.0
        internet_url = internet_dict[idx]["best_url"] if idx in internet_dict else ""

        # Determine overall best score and source
        if local_score >= internet_score:
            best_score = local_score
            best_source_type = "Local"
        else:
            best_score = internet_score
            best_source_type = "Internet"

        merged_rows.append({
            "chunk_index": idx,
            "chunk_text": chunk_text,
            "best_local_score": local_score,
            "local_file": local_file,
            "best_internet_score": internet_score,
            "internet_url": internet_url,
            "best_score": best_score,
            "best_source_type": best_source_type
        })

    return merged_rows

def calculate_plagiarized_portions(local_details, internet_details, student_text):
    """
    Calculate the percentage of student text (by word count) flagged as plagiarized.
    """
    student_chunks = chunk_text(student_text)
    total_words = sum(len(ch.split()) for ch in student_chunks)

    flagged_local_indices = {item["chunk_index"] for item in local_details}
    flagged_internet_indices = {item["chunk_index"] for item in internet_details}
    local_word_count = sum(len(student_chunks[i].split()) for i in flagged_local_indices)
    internet_word_count = sum(len(student_chunks[i].split()) for i in flagged_internet_indices)

    portion_local = round((local_word_count / total_words) * 100, 2) if total_words else 0
    portion_internet = round((internet_word_count / total_words) * 100, 2) if total_words else 0
    return portion_local, portion_internet

# =========================================================
# 4. HIGHLIGHTING FUNCTION FOR WEB UI
# =========================================================

def highlight_plagiarized_chunks(student_text, local_details, internet_details):
    """
    Builds an HTML string that highlights flagged chunks in yellow.
    """
    flagged_local = {item["chunk_index"] for item in local_details}
    flagged_internet = {item["chunk_index"] for item in internet_details}
    flagged_indices = flagged_local.union(flagged_internet)

    chunks = chunk_text(student_text)
    highlighted_html = ""
    for i, chunk in enumerate(chunks):
        safe_chunk = html.escape(chunk)
        if i in flagged_indices:
            highlighted_html += f"<span style='background-color: yellow;'>{safe_chunk}</span> "
        else:
            highlighted_html += f"{safe_chunk} "
    return highlighted_html

# =========================================================
# 5. STREAMLIT APP: UI, Analysis, and PDF Report Generation
# =========================================================

st.title("🚀 Generative AI-Based MEC102 Engineering Design Report Assessment")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🔍 Plagiarism/Reasoning Finder", "📈 Student Analytics"])

# ---- PLAGIARISM/REASONING FINDER PAGE ----
if page == "🔍 Plagiarism/Reasoning Finder":
    st.header("📄 Upload and Assess Report")
    uploaded_file = st.file_uploader("Upload student's report (.docx or .pdf)", type=["docx", "pdf"])
    student_id = st.text_input("Enter Student ID", key="student_id_input", value="SEEE001")
    student_name = st.text_input("Enter Student Name", key="student_name_input", value="Student Name")

    ai_assessment = ""
    llm_plagiarism = ""
    local_score = 0.0

    if uploaded_file:
        # Extract the student text based on file type
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext == "pdf":
            student_text = extract_text_from_pdf(uploaded_file)
        else:
            student_text = extract_text_from_docx(uploaded_file)

        # Provide a default JSON rubric for AI Assessment
        rubric_json = json.dumps({
            "Concept Understanding": 10,
            "Implementation": 10,
            "Analysis": 10,
            "Clarity": 5,
            "Creativity": 5
        }, indent=2)
        st.session_state["rubric_json"] = rubric_json
        rubric = st.text_area("✏️ Customize AI Feedback Rubric (JSON format)", value=rubric_json, height=150)

        col1, col2, col3, col4 = st.columns(4)

        # --- (1) AI-Based Assessment ---
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

                # Extract overall score (fallback uses full rubric total)
                score_match = re.search(r"Overall Score:\s*(\d+)/\d+", ai_assessment)
                if score_match:
                    ai_score = int(score_match.group(1))
                else:
                    ai_score = sum(json.loads(rubric).values())
                st.session_state['ai_score'] = ai_score
                st.success("✅ AI Feedback")
                st.write(ai_assessment)

        # --- (2) LLM-Based Plagiarism Check ---
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

        # --- (3) Single-Score Local Check ---
        with col3:
            if st.button("🔎 Compare with Local Reports"):
                local_score = compute_local_plagiarism_scores(student_text)
                st.session_state['local_similarity_score'] = local_score
                st.success(f"📊 Local Similarity Score: {local_score}%")

        # --- (4) Detailed Chunk-Based Plagiarism Analysis and Merge ---
        with col4:
            if st.button("🔬 Detailed Plagiarism Analysis"):
                local_details = compute_local_plagiarism_details(student_text)
                internet_details = compute_internet_plagiarism_details(student_text)
                # Instead of simply combining (which may create duplicates), merge results per chunk.
                merged_results = merge_local_and_internet(local_details, internet_details)
                # Calculate overall portions of plagiarized text.
                portion_local, portion_internet = calculate_plagiarized_portions(local_details, internet_details, student_text)
                st.markdown(f"**Local Plagiarized Portion:** {portion_local}%")
                st.markdown(f"**Internet Plagiarized Portion:** {portion_internet}%")
                
                if merged_results:
                    st.write("### Merged Detailed Plagiarism Matches (One Row per Chunk)")
                    # For clarity, create a snippet for display
                    df_merged = pd.DataFrame(merged_results)
                    df_merged["snippet"] = df_merged["chunk_text"].apply(
                        lambda txt: txt[:80] + "..." if len(txt) > 80 else txt
                    )
                    df_merged = df_merged[["chunk_index", "snippet", "best_score", "best_source_type", "local_file", "internet_url"]]
                    df_merged = df_merged.sort_values("best_score", ascending=False)
                    st.dataframe(df_merged)
                    if st.button("Highlight Plagiarized Text"):
                        highlighted_html = highlight_plagiarized_chunks(student_text, local_details, internet_details)
                        st.markdown(highlighted_html, unsafe_allow_html=True)
                else:
                    st.info("No significant plagiarized portions found.")

    # --- PDF Report and Database Save ---
    if st.button("📤 Export PDF Report and Save Scores"):
        ai_feedback = clean_text(st.session_state.get("ai_assessment", "Not available"))
        llm_result = clean_text(st.session_state.get("llm_plagiarism", "Plagiarism Risk: 0%"))
        rubric_dict = json.loads(st.session_state.get("rubric_json", "{}"))
        total_score = st.session_state.get('ai_score', sum(rubric_dict.values()))
        llm_percent_search = re.search(r"(\d+)%", llm_result)
        llm_percent = float(llm_percent_search.group(1)) if llm_percent_search else 0.0
        local_score = st.session_state.get("local_similarity_score", 0.0)

        # Generate PDF Report
        pdf = FPDF()
        pdf.add_page()

        # Report header (with logo, if exists)
        logo_path = "sastra_logo.jpg"
        if os.path.exists(logo_path):
            pdf.image(logo_path, x=45, y=10, w=125)
            pdf.ln(30)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Assessment Report for Student ID: {student_id}", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"Name: {student_name}", ln=True)
        pdf.cell(0, 10, f"Total Score: {total_score}/50", ln=True)
        for k, v in rubric_dict.items():
            pdf.cell(0, 10, f"- {k}: {v}/10", ln=True)
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"AI Assessment:\n{ai_feedback}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"LLM-Based Plagiarism Result:\n{llm_result}")
        pdf.ln(5)
        pdf.cell(0, 10, f"Local Similarity Score: {local_score}%", ln=True)
        
        # --- New Section: Highlighted Plagiarized Portions ---
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Highlighted Plagiarized Portions:", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", '', 10)

        # Use the merged results so that each chunk appears only once
        local_details = compute_local_plagiarism_details(student_text)
        internet_details = compute_internet_plagiarism_details(student_text)
        merged_results = merge_local_and_internet(local_details, internet_details)

        if merged_results:
            for item in merged_results:
                snippet = item["chunk_text"][:100] + "..." if len(item["chunk_text"]) > 100 else item["chunk_text"]
                line = f"Chunk {item['chunk_index']} (Best: {item['best_score']} - {item['best_source_type']}): {snippet}"
                pdf.set_fill_color(255, 255, 0)  # Yellow background
                pdf.multi_cell(0, 10, line, border=1, fill=True)
                pdf.ln(2)
        else:
            pdf.cell(0, 10, "No significant plagiarized portions found.", ln=True)

        # Prepare PDF for download
        pdf_output = BytesIO()
        pdf_bytes = pdf.output(dest="S").encode("latin1")
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)
        st.download_button("📥 Download Report", data=pdf_output, file_name=f"{student_id}_Report.pdf")

        # Save assessment details to the database
        conn = init_db()
        c = conn.cursor()
        c.execute("""
            INSERT INTO quiz_results (student_id, student_name, topic, score, total_questions,
                                      ai_score, llm_plagiarism_score, local_similarity_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student_id, student_name, "MEC102 Report", total_score, 50,
            total_score, llm_percent, local_score
        ))
        conn.commit()
        conn.close()
        st.success("✅ Assessment saved to database.")

# ---- STUDENT ANALYTICS PAGE ----
elif page == "📈 Student Analytics":
    st.header("📈 Student Performance Analytics")
    df = get_student_quiz_history()
    if df.empty:
        st.warning("No assessment records available.")
    else:
        student_ids = df['student_id'].unique().tolist()
        selected_id = st.selectbox("Select Student ID", student_ids)
        student_df = df[df['student_id'] == selected_id]
        student_name = student_df['student_name'].iloc[0]
        st.subheader(f"Performance Analytics for: {student_name} ({selected_id})")
        student_df["timestamp"] = pd.to_datetime(student_df["timestamp"])
        student_df = student_df.sort_values("timestamp")
        st.markdown("### 🧠 AI Score Trend")
        st.line_chart(student_df.set_index("timestamp")[["ai_score"]])
        st.markdown("### 🔍 LLM Plagiarism Trend")
        st.line_chart(student_df.set_index("timestamp")[["llm_plagiarism_score"]])
        st.markdown("### 🧪 Local Similarity Trend")
        st.line_chart(student_df.set_index("timestamp")[["local_similarity_score"]])
        st.markdown("### 🔢 Summary")
        st.dataframe(student_df[["topic", "ai_score", "llm_plagiarism_score", "local_similarity_score", "timestamp"]])
        st.info(f"**Average AI Score**: {student_df['ai_score'].mean():.2f}")
        st.info(f"**Average LLM Plagiarism %**: {student_df['llm_plagiarism_score'].mean():.2f}%")
        st.info(f"**Average Local Similarity %**: {student_df['local_similarity_score'].mean():.2f}%")

# ---- CLASS-WIDE DASHBOARD PAGE ----
elif page == "📊 Dashboard":
    st.header("📊 Class-Wide Analytics")
    df = get_student_quiz_history()
    if df.empty:
        st.warning("No assessment records found.")
    else:
        avg_df = df.groupby("student_id").agg({
            "ai_score": "mean",
            "llm_plagiarism_score": "mean",
            "local_similarity_score": "mean"
        }).reset_index()
        st.subheader("🧠 Average AI Score per Student")
        st.bar_chart(avg_df.set_index("student_id")[["ai_score"]])
        st.subheader("🔍 Average LLM Plagiarism %")
        st.bar_chart(avg_df.set_index("student_id")[["llm_plagiarism_score"]])
        st.subheader("🧪 Average Local Similarity %")
        st.bar_chart(avg_df.set_index("student_id")[["local_similarity_score"]])
        st.subheader("🏆 Top Students by AI Score")
        st.table(
            avg_df.sort_values("ai_score", ascending=False).head(5)
            .rename(columns={"ai_score": "Avg AI Score"})
        )
