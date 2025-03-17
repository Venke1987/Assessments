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

# ---------------- NEW IMPORTS (PLACEHOLDERS) ----------------
# You may need to install these libraries if you actually want to use them
# !pip install opencv-python face_recognition xgboost scikit-learn
try:
    import cv2  # for Computer Vision (OMR, Face/Eye detection, etc.)
except ImportError:
    pass

try:
    import face_recognition  # for facial recognition
except ImportError:
    pass

try:
    import xgboost as xgb  # for XGBoost-based predictions
except ImportError:
    pass

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    pass

load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # Use Streamlit Secrets API Key

# ---------------- DATABASE INIT ----------------
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

def save_quiz_result(student_id, student_name, topic, score, total_questions):
    conn = init_db()
    c = conn.cursor()
    c.execute('''
    INSERT INTO quiz_results (student_id, student_name, topic, score, total_questions, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (student_id, student_name, topic, score, total_questions, datetime.now()))
    conn.commit()
    conn.close()

def get_student_quiz_history(student_id=None):
    conn = init_db()
    c = conn.cursor()
    
    if student_id:
        c.execute('''
        SELECT student_id, student_name, topic, score, total_questions, timestamp
        FROM quiz_results
        WHERE student_id = ?
        ORDER BY timestamp DESC
        ''', (student_id,))
    else:
        c.execute('''
        SELECT student_id, student_name, topic, score, total_questions, timestamp
        FROM quiz_results
        ORDER BY timestamp DESC
        ''')
    
    results = c.fetchall()
    conn.close()
    
    if results:
        return pd.DataFrame(results, columns=['student_id', 'student_name', 'topic', 'score', 'total_questions', 'timestamp'])
    return pd.DataFrame()

def extract_text_from_docx(uploaded_file):
    """Extract text from a .docx file"""
    try:
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def extract_text_from_pdf(uploaded_file):
    """Extract text from a .pdf file."""
    try:
        text = ""
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def check_python_syntax(code):
    """Check for syntax errors in uploaded Python code."""
    try:
        ast.parse(code)
        return None  # No syntax errors
    except SyntaxError as e:
        return f"Syntax Error: {e}"

# ---------------- SAMPLE STUDENT DATA ----------------
students_data = {
    "SEEE001": {"name": "Adhithya V", "proficiency": {"Fuzzy Logic": {1: 0.59, 2: 0.64, 3: 0.69, 4: 0.74, 5: 0.79, 6: 0.84}}},
    "SEEE002": {"name": "Akety Manjunath", "proficiency": {"Data Science": {1: 0.6, 2: 0.65, 3: 0.7, 4: 0.75, 5: 0.8, 6: 0.85}}},
    "SEEE003": {"name": "Aravind S", "proficiency": {"Bayesian Learning": {1: 0.37, 2: 0.42, 3: 0.47, 4: 0.52, 5: 0.57, 6: 0.62}}},
    "SEEE004": {"name": "Aswin S", "proficiency": {"NLP": {1: 0.54, 2: 0.59, 3: 0.64, 4: 0.69, 5: 0.74, 6: 0.79}}},
    "SEEE005": {"name": "Avinash Kumar R", "proficiency": {"Data Science": {1: 0.69, 2: 0.74, 3: 0.79, 4: 0.84, 5: 0.89, 6: 0.94}}},
    "SEEE007": {"name": "Bhavya Sri B", "proficiency": {"IoT": {1: 0.59, 2: 0.64, 3: 0.69, 4: 0.74, 5: 0.79, 6: 0.84}}},
    "SEEE008": {"name": "Challagandla Anantha Pavan", "proficiency": {"Cybersecurity": {1: 0.7, 2: 0.75, 3: 0.8, 4: 0.85, 5: 0.9, 6: 0.95}}},
    "SEEE009": {"name": "Nagadarahas Kumar C S", "proficiency": {"Cybersecurity": {1: 0.6, 2: 0.65, 3: 0.7, 4: 0.75, 5: 0.8, 6: 0.85}}},
    "SEEE010": {"name": "Dodda Sri Pujitha", "proficiency": {"Edge AI": {1: 0.73, 2: 0.78, 3: 0.83, 4: 0.88, 5: 0.93, 6: 0.98}}},
    "SEEE011": {"name": "Dondapati Vallapala Yami", "proficiency": {"Deep Learning": {1: 0.58, 2: 0.63, 3: 0.68, 4: 0.73, 5: 0.78, 6: 0.83}}},
    "SEEE012": {"name": "Guduru Venkata Sai Karthikeya", "proficiency": {"Generative AI": {1: 0.42, 2: 0.47, 3: 0.52, 4: 0.57, 5: 0.62, 6: 0.67}}},
    "SEEE013": {"name": "Hamsitha P", "proficiency": {"AI Ethics": {1: 0.39, 2: 0.44, 3: 0.49, 4: 0.54, 5: 0.59, 6: 0.64}}},
    "SEEE014": {"name": "Harini M", "proficiency": {"Generative AI": {1: 0.5, 2: 0.55, 3: 0.6, 4: 0.65, 5: 0.7, 6: 0.75}}},
    "SEEE015": {"name": "Harrish B", "proficiency": {"Robotics": {1: 0.45, 2: 0.5, 3: 0.55, 4: 0.6, 5: 0.65, 6: 0.7}}},
    "SEEE016": {"name": "Jivan Prasath S", "proficiency": {"Swarm Intelligence": {1: 0.51, 2: 0.56, 3: 0.61, 4: 0.66, 5: 0.71, 6: 0.76}}},
    "SEEE017": {"name": "Jonnalagadda Susrith", "proficiency": {"Quantum Computing": {1: 0.45, 2: 0.5, 3: 0.55, 4: 0.6, 5: 0.65, 6: 0.7}}},
    "SEEE018": {"name": "Kamaleshwar M", "proficiency": {"Data Science": {1: 0.72, 2: 0.77, 3: 0.82, 4: 0.87, 5: 0.92, 6: 0.97}}},
    "SEEE019": {"name": "Karthik Periyakarupphan M", "proficiency": {"NLP": {1: 0.36, 2: 0.41, 3: 0.46, 4: 0.51, 5: 0.56, 6: 0.61}}},
    "SEEE020": {"name": "Kathirazagan V", "proficiency": {"Bayesian Learning": {1: 0.44, 2: 0.49, 3: 0.54, 4: 0.59, 5: 0.64, 6: 0.69}}},
    "SEEE021": {"name": "Kishore R", "proficiency": {"Robotics": {1: 0.72, 2: 0.77, 3: 0.82, 4: 0.87, 5: 0.92, 6: 0.97}}},
    "SEEE022": {"name": "Krishnakumar Aditi J", "proficiency": {"Computer Vision": {1: 0.64, 2: 0.69, 3: 0.74, 4: 0.79, 5: 0.84, 6: 0.89}}},
    "SEEE023": {"name": "Logavarshini K", "proficiency": {"Optimization": {1: 0.7, 2: 0.75, 3: 0.8, 4: 0.85, 5: 0.9, 6: 0.95}}},
    "SEEE024": {"name": "Malapalli Charitha Reddy", "proficiency": {"Deep Learning": {1: 0.54, 2: 0.59, 3: 0.64, 4: 0.69, 5: 0.74, 6: 0.79}}},
    "SEEE025": {"name": "Manas M Nair", "proficiency": {"Bayesian Learning": {1: 0.64, 2: 0.69, 3: 0.74, 4: 0.79, 5: 0.84, 6: 0.89}}},
    "SEEE026": {"name": "Mervath M", "proficiency": {"Robotics": {1: 0.55, 2: 0.6, 3: 0.65, 4: 0.7, 5: 0.75, 6: 0.8}}},
    "SEEE027": {"name": "Nidhish Kumar K", "proficiency": {"Bayesian Learning": {1: 0.37, 2: 0.42, 3: 0.47, 4: 0.52, 5: 0.57, 6: 0.62}}},
    "SEEE028": {"name": "Palapati Jahnavi", "proficiency": {"Reinforcement Learning": {1: 0.65, 2: 0.7, 3: 0.75, 4: 0.8, 5: 0.85, 6: 0.9}}},
    "SEEE029": {"name": "Pola Kaarthi", "proficiency": {"Fuzzy Logic": {1: 0.74, 2: 0.79, 3: 0.84, 4: 0.89, 5: 0.94, 6: 0.99}}},
    "SEEE030": {"name": "Pullela Vaishnavi", "proficiency": {"Data Science": {1: 0.39, 2: 0.44, 3: 0.49, 4: 0.54, 5: 0.59, 6: 0.64}}},
    "SEEE031": {"name": "Raghul T", "proficiency": {"Generative AI": {1: 0.43, 2: 0.48, 3: 0.53, 4: 0.58, 5: 0.63, 6: 0.68}}},
    "SEEE032": {"name": "Rajaprabu K", "proficiency": {"Big Data": {1: 0.41, 2: 0.46, 3: 0.51, 4: 0.56, 5: 0.61, 6: 0.66}}},
    "SEEE033": {"name": "Rishi A", "proficiency": {"NLP": {1: 0.41, 2: 0.46, 3: 0.51, 4: 0.56, 5: 0.61, 6: 0.66}}},
    "SEEE034": {"name": "Rithick Reddy S", "proficiency": {"Swarm Intelligence": {1: 0.48, 2: 0.53, 3: 0.58, 4: 0.63, 5: 0.68, 6: 0.73}}},
    "SEEE035": {"name": "Tera Sai Tejeshwar Reddy", "proficiency": {"Swarm Intelligence": {1: 0.59, 2: 0.64, 3: 0.69, 4: 0.74, 5: 0.79, 6: 0.84}}},
    "SEEE036": {"name": "Saikirthiga R", "proficiency": {"Data Science": {1: 0.55, 2: 0.6, 3: 0.65, 4: 0.7, 5: 0.75, 6: 0.8}}},
    "SEEE037": {"name": "Sangam JayaVardhan Reddy", "proficiency": {"Big Data": {1: 0.38, 2: 0.43, 3: 0.48, 4: 0.53, 5: 0.58, 6: 0.63}}},
    "SEEE038": {"name": "Sarveshwaran S", "proficiency": {"ML Basics": {1: 0.71, 2: 0.76, 3: 0.81, 4: 0.86, 5: 0.91, 6: 0.96}}},
    "SEEE039": {"name": "Shanjay Sundhar S", "proficiency": {"IoT": {1: 0.68, 2: 0.73, 3: 0.78, 4: 0.83, 5: 0.88, 6: 0.93}}},
    "SEEE040": {"name": "Shreeya Kannan", "proficiency": {"Fuzzy Logic": {1: 0.62, 2: 0.67, 3: 0.72, 4: 0.77, 5: 0.82, 6: 0.87}}},
    "SEEE041": {"name": "Shreya S", "proficiency": {"AI Ethics": {1: 0.61, 2: 0.66, 3: 0.71, 4: 0.76, 5: 0.81, 6: 0.86}}},
    "SEEE042": {"name": "Sri Kamal Krishank S", "proficiency": {"Data Science": {1: 0.46, 2: 0.51, 3: 0.56, 4: 0.61, 5: 0.66, 6: 0.71}}},
    "SEEE043": {"name": "Sri Ramana S", "proficiency": {"Quantum Computing": {1: 0.37, 2: 0.42, 3: 0.47, 4: 0.52, 5: 0.57, 6: 0.62}}},
    "SEEE044": {"name": "Subashree S", "proficiency": {"Reinforcement Learning": {1: 0.48, 2: 0.53, 3: 0.58, 4: 0.63, 5: 0.68, 6: 0.73}}},
    "SEEE045": {"name": "Subhiksha N", "proficiency": {"Bayesian Learning": {1: 0.38, 2: 0.43, 3: 0.48, 4: 0.53, 5: 0.58, 6: 0.63}}},
    "SEEE046": {"name": "Sukhi Sudharshan S", "proficiency": {"Robotics": {1: 0.64, 2: 0.69, 3: 0.74, 4: 0.79, 5: 0.84, 6: 0.89}}},
    "SEEE047": {"name": "Thanuja", "proficiency": {"Computer Vision": {1: 0.54, 2: 0.59, 3: 0.64, 4: 0.69, 5: 0.74, 6: 0.79}}},
    "SEEE048": {"name": "Vasantha Kumar A", "proficiency": {"Quantum Computing": {1: 0.5, 2: 0.55, 3: 0.6, 4: 0.65, 5: 0.7, 6: 0.75}}},
    "SEEE049": {"name": "Velmugilan S", "proficiency": {"Embedded Systems": {1: 0.52, 2: 0.57, 3: 0.62, 4: 0.67, 5: 0.72, 6: 0.77}}},
    "SEEE050": {"name": "Velmurugan S", "proficiency": {"Fuzzy Logic": {1: 0.5, 2: 0.55, 3: 0.6, 4: 0.65, 5: 0.7, 6: 0.75}}},
    "SEEE051": {"name": "Viamrsh P S", "proficiency": {"Generative AI": {1: 0.45, 2: 0.5, 3: 0.55, 4: 0.6, 5: 0.65, 6: 0.7}}},
    "SEEE052": {"name": "Vilohitan R", "proficiency": {"Reinforcement Learning": {1: 0.44, 2: 0.49, 3: 0.54, 4: 0.59, 5: 0.64, 6: 0.69}}},
    "SEEE053": {"name": "Vundela Guru Venkata Ajay Kumar Reddy", "proficiency": {"Big Data": {1: 0.72, 2: 0.77, 3: 0.82, 4: 0.87, 5: 0.92, 6: 0.97}}},
    "SEEE054": {"name": "Yadlapalli Madhuri", "proficiency": {"Robotics": {1: 0.44, 2: 0.49, 3: 0.54, 4: 0.59, 5: 0.64, 6: 0.69}}},
    "SEEE055": {"name": "Yohan K", "proficiency": {"Data Science": {1: 0.64, 2: 0.69, 3: 0.74, 4: 0.79, 5: 0.84, 6: 0.89}}},
    "SEEE056": {"name": "Vinupriya A", "proficiency": {"Computer Vision": {1: 0.45, 2: 0.5, 3: 0.55, 4: 0.6, 5: 0.65, 6: 0.7}}},
    "SEEE057": {"name": "Aparajita S", "proficiency": {"Computer Vision": {1: 0.73, 2: 0.78, 3: 0.83, 4: 0.88, 5: 0.93, 6: 0.98}}},
    "SEEE058": {"name": "Sriram L", "proficiency": {"NLP": {1: 0.63, 2: 0.68, 3: 0.73, 4: 0.78, 5: 0.83, 6: 0.88}}},
    "SEEE059": {"name": "Nandhini S", "proficiency": {"Deep Learning": {1: 0.55, 2: 0.6, 3: 0.65, 4: 0.7, 5: 0.75, 6: 0.8}}},
    "SEEE060": {"name": "Sreenath G", "proficiency": {"IoT": {1: 0.52, 2: 0.57, 3: 0.62, 4: 0.67, 5: 0.72, 6: 0.77}}},
    "SEEE061": {"name": "Swaran V", "proficiency": {"AI Ethics": {1: 0.54, 2: 0.59, 3: 0.64, 4: 0.69, 5: 0.74, 6: 0.79}}},
    "SEEE061": {"name": "Swaran V", "proficiency": {"AI Ethics": {1: 0.54, 2: 0.59, 3: 0.64, 4: 0.69, 5: 0.74, 6: 0.79}}}
}

# ---------------- SAMPLE QUESTIONS DATA ----------------
questions = [
    {"text": "What is the primary goal of supervised learning?", "options": [
        "To cluster data points without labels",
        "To learn patterns from labeled data to make predictions",
        "To reduce the dimensionality of data",
        "To generate new data samples"
    ], "correct": "To learn patterns from labeled data to make predictions"},
    {"text": "Which algorithm is used for classification?", "options": [
        "K-Means", "Decision Tree", "PCA", "Apriori"
    ], "correct": "Decision Tree"},
    {"text": "What does CNN stand for?", "options": [
        "Convolutional Neural Network", "Central Neural Network",
        "Computational Node Network", "Convolutional Node Network"
    ], "correct": "Convolutional Neural Network"},
    {"text": "Which technique helps reduce overfitting?", "options": [
        "Regularization", "Dropout", "Both 1 & 2", "None"
    ], "correct": "Both 1 & 2"}
]

# ---------------- SESSION STATE INIT ----------------
if "quiz_active" not in st.session_state:
    st.session_state.quiz_active = False
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "total_questions" not in st.session_state:
    st.session_state.total_questions = 5
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "score" not in st.session_state:
    st.session_state.score = 0

# Initialize database
init_db()

# ---------------- STREAMLIT UI LAYOUT ----------------
st.title("🚀 Generative AI-Based Students Assessment System")
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "📊 Dashboard",
        "📝 Take Quiz",
        "📚 Quiz History",
        "📖 AI-Powered Storytelling",
        "🧠 AI-Powered Hints",
        "🔍 AI Peer Assessment",
        "🔍 Plagiarism/Reasoning Finder",
        "📂 Code Evaluation & Plagiarism Check",
        "📄 AI-Based LOR Generator",
        # --- NEW FEATURES FROM THE DIAGRAM ---
        "📝 NLP Answer Evaluation",
        "📝 OMR MCQ Grading",
        "📈 ML Performance Tracking & Prediction",
        "🔒 AI Proctoring & Integrity Checks",
    ]
)

# ---------------- PAGE: DASHBOARD ----------------
if page == "📊 Dashboard":
    st.header("📊 Class Performance Dashboard")
    
    # Select Student
    student_id = st.selectbox("Select a Student", list(students_data.keys()))
    student = students_data[student_id]
    st.subheader(f"Student: {student['name']}")
    
    # Proficiency Data
    prof_data = student["proficiency"]
    topics = list(prof_data.keys())
    bloom_levels = [1, 2, 3, 4, 5, 6]
    
    # Heatmap Data
    heatmap_data = pd.DataFrame(
        {topic: [prof_data[topic].get(level, 0.5) for level in bloom_levels] for topic in topics},
        index=[f"Level {lvl}" for lvl in bloom_levels]
    )
    
    st.subheader("📌 Bloom's Taxonomy Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)
    
    st.subheader("📈 Student Progress Over Time")
    dates = [datetime.now().replace(day=i) for i in range(1, 11)]
    scores = [random.uniform(0.4, 0.9) for _ in range(10)]
    plt.figure(figsize=(8, 5))
    plt.plot(dates, scores, marker='o', linestyle='-')
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Proficiency Score")
    plt.title("Student Progress")
    st.pyplot(plt)

# ---------------- PAGE: AI-POWERED STORYTELLING ----------------
elif page == "📖 AI-Powered Storytelling":
    st.header("📖 AI-Powered Conversational Storytelling")

    user_topic = st.text_input("Enter a topic for the story (e.g., AI in Space, Lost Treasure, Cybersecurity):")

    if "story_conversation" not in st.session_state:
        st.session_state.story_conversation = []

    if st.button("Generate Story"):
        if user_topic.strip():
            prompt = (
                f"You are a friendly AI storyteller. Create a fun and engaging story based on the topic: '{user_topic}'. "
                f"Format the story like a chatbot conversation where an AI and a human character interact."
            )

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a storytelling AI."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=400
            )

            story_text = response.choices[0].message.content.strip()
            st.session_state.story_conversation.append({"role": "AI Storyteller", "content": story_text})

    for message in st.session_state.story_conversation:
        if message["role"] == "AI Storyteller":
            st.markdown(f"**🤖 AI Storyteller:** {message['content']}")

    user_reply = st.text_input("Continue the story... (Optional)")

    if st.button("Continue"):
        if user_reply.strip():
            follow_up_prompt = (
                f"Continue the story based on the user's response: '{user_reply}'. "
                f"Keep the conversational tone between the AI storyteller and the characters."
            )

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a storytelling AI."},
                    {"role": "user", "content": follow_up_prompt}
                ],
                temperature=0.8,
                max_tokens=300
            )

            follow_up_story = response.choices[0].message.content.strip()

            st.session_state.story_conversation.append({"role": "User", "content": user_reply})
            st.session_state.story_conversation.append({"role": "AI Storyteller", "content": follow_up_story})
            st.experimental_rerun()

# ---------------- PAGE: TAKE QUIZ ----------------
elif page == "📝 Take Quiz":
    st.header("📝 Adaptive Quiz")
    
    student_id = st.selectbox("Select your Student ID", list(students_data.keys()))
    st.write(f"Student Name: {students_data[student_id]['name']}")
    
    quiz_topic = st.selectbox(
        "Select a topic for your quiz:",
        ["Generative AI", "Machine Learning", "Reinforcement Learning", "Deep Learning", "Natural Language Processing", "Computer Vision"]
    )
    
    st.session_state.total_questions = st.slider("Number of Questions", min_value=1, max_value=10, value=5)
    
    if not st.session_state.quiz_active:
        if st.button("Start Quiz"):
            st.session_state.quiz_active = True
            st.session_state.question_count = 0
            st.session_state.score = 0

            prompt = (
                f"Create a multiple-choice question about {quiz_topic} with 4 options and indicate the correct answer. "
                f"Format your response as JSON with fields: 'text', 'options' (array of 4), and 'correct'."
            )
            
            with st.spinner("Generating question..."):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an AI that generates educational quiz questions. Return valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=250
                )
                
                try:
                    response_text = response.choices[0].message.content.strip()
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0].strip()
                    
                    st.session_state.current_question = json.loads(response_text)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error generating question: {str(e)}")
                    st.write("Response received:", response.choices[0].message.content)
                    st.session_state.quiz_active = False

    if st.session_state.quiz_active and st.session_state.question_count < st.session_state.total_questions:
        q = st.session_state.current_question
        st.write(f"**Q{st.session_state.question_count+1}:** {q['text']}")
        answer = st.radio("Choose the correct answer:", q["options"], index=None)
        
        if st.button("Submit Answer"):
            if answer:
                if answer == q["correct"]:
                    st.success("✅ Correct Answer!")
                    st.session_state.score += 1
                else:
                    st.error("❌ Incorrect Answer!")
                    st.info(f"The correct answer was: {q['correct']}")
                
                st.session_state.question_count += 1
                
                if st.session_state.question_count < st.session_state.total_questions:
                    prompt = (
                        f"Create a multiple-choice question about {quiz_topic} with 4 options and indicate the correct answer. "
                        f"Format as JSON: 'text', 'options', 'correct'."
                    )
                    
                    with st.spinner("Generating next question..."):
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are an AI that generates educational quiz questions. Return valid JSON only."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                            max_tokens=250
                        )
                        
                        try:
                            response_text = response.choices[0].message.content.strip()
                            if "```json" in response_text:
                                response_text = response_text.split("```json")[1].split("```")[0].strip()
                            elif "```" in response_text:
                                response_text = response_text.split("```")[1].split("```")[0].strip()
                            
                            st.session_state.current_question = json.loads(response_text)
                        except Exception as e:
                            st.error(f"Error generating question: {str(e)}")
                            st.write("Response received:", response.choices[0].message.content)
                            st.session_state.current_question = {
                                "text": "What is a primary advantage of using neural networks?",
                                "options": [
                                    "They always require less data than other models",
                                    "They can automatically learn complex patterns from data",
                                    "They never overfit",
                                    "They always train quickly"
                                ],
                                "correct": "They can automatically learn complex patterns from data"
                            }
                
                st.experimental_rerun()

    elif st.session_state.quiz_active and st.session_state.question_count >= st.session_state.total_questions:
        st.success(f"Quiz Complete! Your Final Score: {st.session_state.score}/{st.session_state.total_questions}")
        
        save_quiz_result(
            student_id=student_id,
            student_name=students_data[student_id]["name"],
            topic=quiz_topic,
            score=st.session_state.score,
            total_questions=st.session_state.total_questions
        )
        
        score_percentage = (st.session_state.score / st.session_state.total_questions) * 100
        if score_percentage >= 80:
            st.balloons()
            st.write("🌟 Excellent! You have a strong understanding of this topic.")
        elif score_percentage >= 60:
            st.write("👍 Good job! You have a solid grasp of the basics.")
        else:
            st.write("📚 Keep learning! Consider reviewing this topic more thoroughly.")
        
        if st.button("Restart Quiz"):
            st.session_state.quiz_active = False
            st.experimental_rerun()

# ---------------- PAGE: QUIZ HISTORY ----------------
elif page == "📚 Quiz History":
    st.header("📚 Quiz History Dashboard")
    
    col1, col2 = st.columns(2)
    with col1:
        view_option = st.radio("View", ["All Students", "Specific Student"])
    
    student_id = None
    if view_option == "Specific Student":
        with col2:
            student_id = st.selectbox("Select Student", list(students_data.keys()))
            st.write(f"Student: {students_data[student_id]['name']}")
    
    history_df = get_student_quiz_history(student_id)
    
    if not history_df.empty:
        history_df['percentage'] = (history_df['score'] / history_df['total_questions'] * 100).round(1)
        
        st.subheader("📋 Quiz Results")
        st.dataframe(history_df[['student_name', 'topic', 'score', 'total_questions', 'percentage', 'timestamp']])
        
        st.subheader("📊 Performance by Topic")
        topic_df = history_df.groupby('topic')[['percentage']].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='topic', y='percentage', data=topic_df, ax=ax)
        plt.title("Average Score by Topic")
        plt.ylabel("Average Score (%)")
        plt.xlabel("Topic")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        if student_id:
            st.subheader("📈 Performance Over Time")
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.sort_values('timestamp')
            
            plt.figure(figsize=(10, 6))
            plt.plot(history_df['timestamp'], history_df['percentage'], marker='o', linestyle='-')
            plt.title(f"Performance Over Time - {students_data[student_id]['name']}")
            plt.ylabel("Score (%)")
            plt.xlabel("Date")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            st.pyplot(plt)
    else:
        st.info("No quiz history available. Take a quiz to see results here.")

# ---------------- PAGE: AI-POWERED HINTS ----------------
elif page == "🧠 AI-Powered Hints":
    st.header("🧠 Get AI-Powered Hints")
    question_text = st.text_area("Enter your question:")
    
    if st.button("Get Hint"):
        prompt = f"Provide a hint for the following question: {question_text}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI assistant providing hints."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=100
        )
        hint = response.choices[0].message.content.strip()
        st.info(f"💡 Hint: {hint}")

# ---------------- PAGE: AI PEER ASSESSMENT ----------------
elif page == "🔍 AI Peer Assessment":
    st.header("🔍 AI-Powered Peer Assessment")
    student_submission = st.text_area("Paste student submission:")
    rubric = {"Concept Understanding": 10, "Implementation": 10, "Analysis": 10, "Clarity": 5, "Creativity": 5}
    
    if st.button("Generate Peer Review"):
        prompt = f"Assess the following submission using this rubric: {json.dumps(rubric)}\n\nSubmission: {student_submission}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI generating structured peer assessments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        review = response.choices[0].message.content.strip()
        st.success("✅ AI Peer Review Generated:")
        st.write(review)

# ---------------- PAGE: PLAGIARISM / REASONING FINDER ----------------
elif page == "🔍 Plagiarism/Reasoning Finder":
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

    col1, col2 = st.columns(2)
    
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
                st.success("✅ AI Feedback Generated:")
                st.write(review)
            else:
                st.warning("⚠️ Please upload a valid document.")
    
    with col2:
        if st.button("🔍 Check for Plagiarism"):
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
                            {
                                "role": "system",
                                "content": (
                                    "You are an AI that checks for plagiarism in academic work. "
                                    "Always provide a clear plagiarism percentage in your first line."
                                )
                            },
                            {"role": "user", "content": plagiarism_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=400
                    )
                    plagiarism_result = plagiarism_response.choices[0].message.content.strip()
                    
                    try:
                        if "Plagiarism Risk:" in plagiarism_result:
                            first_line = plagiarism_result.split('\n')[0]
                            percentage_text = first_line.split('Plagiarism Risk:')[1].strip()
                            if '%' in percentage_text:
                                percentage = float(percentage_text.replace('%', '').strip())
                                st.subheader("Plagiarism Risk Score:")
                                st.progress(percentage / 100)
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

                    # Example code to store the result in DB if needed
                    # (Requires "current_student_id" or some mechanism)
            else:
                st.warning("⚠️ Please upload a valid document.")

# ---------------- PAGE: CODE EVALUATION & PLAGIARISM CHECK ----------------
elif page == "📂 Code Evaluation & Plagiarism Check":
    st.header("📂 Code Evaluation & Plagiarism Check")

    uploaded_code = st.file_uploader("Upload a Python (.py) file", type=["py"])
    
    if uploaded_code is not None:
        code_content = uploaded_code.getvalue().decode("utf-8")
        
        st.subheader("📜 Uploaded Code:")
        st.code(code_content, language="python")

        syntax_error = check_python_syntax(code_content)
        if syntax_error:
            st.error(syntax_error)
        else:
            st.success("✅ No syntax errors found!")

        if st.button("📊 Evaluate Code & Check Plagiarism"):
            with st.spinner("Analyzing Code..."):
                prompt = (
                    "Analyze the following Python program. Provide:\n"
                    "1. Correctness Score\n2. Efficiency Score\n3. Readability Score\n"
                    "4. Plagiarism Risk Score\n5. Suggestions for Improvement\n\n"
                    f"Code:\n{code_content}"
                )
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an AI that evaluates Python code."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=600
                )
                evaluation_result = response.choices[0].message.content.strip()
                st.success("✅ AI Evaluation Generated:")
                st.write(evaluation_result)

        if st.button("🔍 Check for Plagiarism"):
            with st.spinner("Checking plagiarism..."):
                plagiarism_prompt = (
                    f"Analyze the following Python code for plagiarism risk. "
                    f"Your response MUST include a clear plagiarism percentage in the first line with this exact format: "
                    f"'Plagiarism Risk: XX%' (where XX is 0-100). Then provide explanation below.\n\n"
                    f"Code:\n{code_content}"
                )
                plagiarism_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an AI that checks for plagiarism in Python code. "
                                "Always provide a clear plagiarism percentage in your first line."
                            )
                        },
                        {"role": "user", "content": plagiarism_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400
                )
                plagiarism_result = plagiarism_response.choices[0].message.content.strip()
                
                try:
                    if "Plagiarism Risk:" in plagiarism_result:
                        first_line = plagiarism_result.split('\n')[0]
                        percentage_text = first_line.split('Plagiarism Risk:')[1].strip()
                        if '%' in percentage_text:
                            percentage = float(percentage_text.replace('%', '').strip())
                            st.subheader("Plagiarism Risk Score:")
                            st.progress(percentage / 100)
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

# ---------------- PAGE: AI-BASED LOR GENERATOR ----------------
elif page == "📄 AI-Based LOR Generator":
    st.header("📄 AI-Powered Letter of Recommendation (LOR) Generator")

    student_id = st.selectbox("Select Student ID", list(students_data.keys()))
    student = students_data[student_id]

    st.subheader(
        f"Student: {student['name']} | {student.get('department', 'B.Tech. Robotics & AI')}, "
        f"{student.get('year', 'Third Year')}"
    )
    st.write(f"**CGPA:** {student.get('cgpa', 'Not Available')}")

    lor_purpose = st.radio("Purpose of LOR", ["Higher Studies", "Internship", "Job Application"])
    skills = st.text_area("Enter key skills (comma-separated)", "Machine Learning, Research, Leadership")
    achievements = st.text_area("Enter achievements (comma-separated)", "Won AI Hackathon, Published IEEE Paper")

    if st.button("Generate LOR"):
        with st.spinner("Generating Letter of Recommendation..."):
            prompt = f"""
            Write a professional Letter of Recommendation (LOR) for a student. 
            The student details are:
            - Name: {student['name']}
            - Department: {student.get('department', 'B.Tech. Robotics & AI, Third Year')}
            - CGPA: {student.get('cgpa', 'Not Available')}
            - Purpose: {lor_purpose if lor_purpose else "Not Provided"}
            - Skills: {skills if skills else "Not Provided"}
            - Achievements: {achievements if achievements else "Not Provided"}

            The LOR should be formal, structured, and highlight the student's strengths, 
            academic achievements, and suitability for {lor_purpose}.
            """

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an AI that writes professional letters of recommendation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=400
            )
            st.session_state.lor_text = response.choices[0].message.content.strip()
            st.success("✅ Letter of Recommendation Generated!")
            st.text_area("Generated LOR", st.session_state.lor_text, height=300)

    if "lor_text" in st.session_state:
        refine_prompt = st.text_area("🔄 Enter improvements to refine the LOR (optional)")

        if st.button("Refine LOR"):
            with st.spinner("Refining LOR..."):
                refine_request = f"Refine and improve the following LOR based on this feedback: {refine_prompt}\n\n{st.session_state.lor_text}"
                refined_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You refine and improve letters of recommendation based on feedback."},
                        {"role": "user", "content": refine_request}
                    ],
                    temperature=0.6,
                    max_tokens=400
                )
                st.session_state.lor_text = refined_response.choices[0].message.content.strip()
                st.success("✅ LOR Refined!")
                st.text_area("Refined LOR", st.session_state.lor_text, height=300)

        doc = docx.Document()
        doc.add_paragraph(st.session_state.lor_text)
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        st.download_button(
            label="📥 Download LOR (Word)",
            data=buffer,
            file_name="LOR.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

# ---------------------------------------------------------------------------
# ---------------- NEW SECTIONS FROM THE DIAGRAM ----------------------------
# ---------------------------------------------------------------------------

# ---------------- PAGE: NLP ANSWER EVALUATION ----------------
elif page == "📝 NLP Answer Evaluation":
    """
    This section demonstrates how to evaluate open-ended answers using NLP (GPT-4o).
    For example, you can ask the user to input a question and their written answer, 
    and the system will provide feedback or a grade.
    """
    st.header("📝 NLP Answer Evaluation")
    question_prompt = st.text_area("Question/Prompt for the student:")
    student_answer = st.text_area("Student's Answer:")

    if st.button("Evaluate Answer"):
        if question_prompt.strip() and student_answer.strip():
            # Construct an NLP-based evaluation prompt
            eval_prompt = (
                f"Evaluate the student's answer to the following question:\n\n"
                f"Question: {question_prompt}\n\n"
                f"Student Answer: {student_answer}\n\n"
                f"Provide a short evaluation of correctness, clarity, and completeness. "
                f"Then provide a final grade (1-10)."
            )
            with st.spinner("Evaluating via GPT-4o..."):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an AI specialized in NLP-based short-answer evaluation."},
                        {"role": "user", "content": eval_prompt}
                    ],
                    temperature=0.4,
                    max_tokens=250
                )
                evaluation = response.choices[0].message.content.strip()
            st.success("✅ Evaluation Complete:")
            st.write(evaluation)
        else:
            st.warning("Please provide both a question and a student answer.")

# ---------------- PAGE: COMPUTER VISION OMR GRADING ----------------
elif page == "📝 OMR MCQ Grading":
    """
    This section shows a placeholder for scanning OMR (Optical Mark Recognition) sheets
    to automatically grade multiple-choice responses from scanned images.
    """
    st.header("📝 Computer Vision-based OMR MCQ Grading")

    uploaded_omr = st.file_uploader("Upload a scanned OMR sheet (image)", type=["png", "jpg", "jpeg"])
    if uploaded_omr is not None:
        file_bytes = np.asarray(bytearray(uploaded_omr.read()), dtype=np.uint8)
        try:
            # Convert to OpenCV image
            cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv_image, caption="Uploaded OMR Sheet", use_column_width=True)
            
            if st.button("Process OMR"):
                st.info("Processing OMR... (placeholder)")
                # TODO: Add actual OMR detection logic (circles detection, etc.)
                # Example pseudo-code:
                # 1. Convert to grayscale
                # 2. Thresholding or morphological operations
                # 3. Detect marked bubbles
                # 4. Map them to question numbers and options
                # 5. Compare with answer key
                st.success("OMR processing complete. Placeholder result: Score = 8/10.")
        except Exception as e:
            st.error(f"Error reading image: {str(e)}")
    else:
        st.info("Upload an OMR image to begin.")

# ---------------- PAGE: ML PERFORMANCE TRACKING & PREDICTION ----------------
elif page == "📈 ML Performance Tracking & Prediction":
    """
    Demonstrates how you might track student performance over time and predict future performance
    using ML models like Random Forest or XGBoost.
    """
    st.header("📈 ML Performance Tracking & Prediction")

    st.write("This page illustrates how to use ML (Random Forest/XGBoost) to predict future performance.")
    st.markdown(
        """
        **Steps (conceptual):**
        1. Collect student performance data over time (scores, quiz results, etc.).
        2. Prepare features (e.g., recent quiz scores, attendance, etc.).
        3. Train a model (RandomForest/XGBoost) to predict future exam performance.
        4. Provide early interventions if the predicted performance is below a threshold.
        """
    )

    # Placeholder data
    if st.button("Train Sample Model"):
        # Just a placeholder to show how you might train
        # real data and libraries needed
        st.info("Training placeholder RandomForest model on synthetic data...")

        try:
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score

            X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            rf = RandomForestClassifier(n_estimators=10, random_state=42)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.success(f"Sample RandomForest Accuracy: {acc*100:.2f}%")
        except Exception as e:
            st.error("Scikit-learn not installed or an error occurred:")
            st.error(str(e))

    st.write("Similarly, you could use XGBoost for performance prediction:")
    if st.button("Train XGBoost Model"):
        try:
            from sklearn.datasets import make_regression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score

            X, y = make_regression(n_samples=100, n_features=5, n_informative=3, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Placeholder XGBoost regressor
            model = xgb.XGBRegressor(n_estimators=10, use_label_encoder=False)
            model.fit(X_train, y_train, eval_metric='rmse', verbose=False)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)
            st.success(f"Sample XGBoost R2 Score: {score:.2f}")
        except Exception as e:
            st.error("XGBoost not installed or an error occurred:")
            st.error(str(e))

    st.info("Use the trained model to predict future performance and intervene early if needed.")

# ---------------- PAGE: AI-POWERED PROCTORING & INTEGRITY CHECKS ----------------
elif page == "🔒 AI Proctoring & Integrity Checks":
    """
    Demonstrates how you might implement AI-powered proctoring:
    - Facial Recognition to verify identity
    - Eye-Tracking to ensure the student isn't looking away excessively
    - Behavior Analysis for suspicious activities
    """
    st.header("🔒 AI-Powered Proctoring & Integrity Checks")

    st.write("Below is a **conceptual** placeholder for real-time proctoring features.")
    st.markdown(
        """
        **Potential Approaches:**
        - **Facial Recognition**: Confirm the student's identity matches the ID photo.
        - **Eye-Tracking**: Check if eyes deviate from screen for suspicious durations.
        - **Behavior Analysis**: Detect multiple faces, unusual movements, etc.
        """
    )

    # Placeholder face recognition
    face_image = st.file_uploader("Upload a photo for face verification", type=["png", "jpg", "jpeg"])
    if face_image is not None:
        file_bytes = np.asarray(bytearray(face_image.read()), dtype=np.uint8)
        try:
            cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv_image, caption="Uploaded Face Image", use_column_width=True)

            if st.button("Run Face Recognition"):
                st.info("Performing face recognition... (placeholder)")
                # You would load a known face embedding, compare with uploaded image, etc.
                # e.g. face_recognition.compare_faces(known_encodings, face_encoding)
                st.success("Identity verified with high confidence. (Placeholder result)")
        except Exception as e:
            st.error(f"Error reading image: {str(e)}")

    # Placeholder for eye tracking or behavior analysis
    st.write("For advanced proctoring (eye tracking, behavior analysis), you'd typically need a live camera feed.")
    st.write("Below is just a placeholder demonstration.")

    if st.button("Start Behavior Analysis"):
        st.info("Analyzing user behavior from webcam feed... (placeholder)")
        st.success("No suspicious behavior detected. (Placeholder)")

    st.warning("Note: Real-time proctoring requires additional setup (webcam access, advanced CV).")

