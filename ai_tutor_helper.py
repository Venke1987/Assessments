# ai_tutor_helper.py
#
# This file contains a simplified interface for integrating the AI Tutor
# into the main assessment system application.

import streamlit as st
from datetime import datetime
import json
import openai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def setup_ai_tutor():
    """
    Initialize the AI Tutor - call this function at the start of your application
    to set up the necessary session state variables for the AI Tutor.
    """
    # Initialize session states for AI Tutor
    if "tutor_messages" not in st.session_state:
        st.session_state.tutor_messages = [
            {"role": "assistant", "content": "👋 Hello! I'm your AI Tutor. How can I help you learn today?"}
        ]
    
    if "learning_goals" not in st.session_state:
        st.session_state.learning_goals = []
    
    if "learning_style" not in st.session_state:
        st.session_state.learning_style = "Visual"
        
    if "difficulty_level" not in st.session_state:
        st.session_state.difficulty_level = "Intermediate"
    
    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = datetime.now()
    
    if "tutor_topic" not in st.session_state:
        st.session_state.tutor_topic = "Computer Science"
    
    if "ai_tutor_quiz" not in st.session_state:
        st.session_state.ai_tutor_quiz = []
    
    if "ai_tutor_quiz_score" not in st.session_state:
        st.session_state.ai_tutor_quiz_score = 0
    
    if "ai_tutor_current_question" not in st.session_state:
        st.session_state.ai_tutor_current_question = 0
    
    if "ai_tutor_quiz_started" not in st.session_state:
        st.session_state.ai_tutor_quiz_started = False
    
    # Resource-related states
    if "resource_topic" not in st.session_state:
        st.session_state.resource_topic = "Computer Science"
    
    if "current_resources" not in st.session_state:
        st.session_state.current_resources = []
    
    if "current_resources_key" not in st.session_state:
        st.session_state.current_resources_key = ""
    
    if "saved_resources" not in st.session_state:
        st.session_state.saved_resources = []
    
    # Add states for learning analytics
    if "learning_analytics" not in st.session_state:
        st.session_state.learning_analytics = {
            "history": [],
            "topics": {},
            "activities": {
                "tutoring": 0,
                "practice": 0,
                "quizzes": 0,
                "flashcards": 0,
                "study_notes": 0,
                "resources": 0,
                "learning_plan": 0
            },
            "total_time": 0
        }
    
    if "topic_mastery" not in st.session_state:
        st.session_state.topic_mastery = {
            "concepts": ["Variables", "Functions", "Loops", "Classes", "Algorithms"],
            "scores": [0.8, 0.6, 0.7, 0.4, 0.3]
        }
    
    if "learning_time" not in st.session_state:
        st.session_state.learning_time = {
            "total_minutes": 0,
            "sessions": 0,
            "last_session": datetime.now()
        }
    
    if "learning_plan_goals" not in st.session_state:
        st.session_state.learning_plan_goals = []
    
    if "ai_tutor_initialized" not in st.session_state:
        st.session_state.ai_tutor_initialized = False
    
    if "use_alternative_camera" not in st.session_state:
        st.session_state.use_alternative_camera = False
    
    if "current_learning_plan" not in st.session_state:
        st.session_state.current_learning_plan = None

def display_ai_tutor_page(students_data=None):
    """
    Display the AI Tutor page with proper integration
    
    Args:
        students_data (dict): Dictionary containing student information
    """
    # Make sure AI Tutor is set up
    setup_ai_tutor()
    
    # Create title
    st.title("👨‍🏫 AI Tutor - Interactive Learning Assistant")
    
    # If no student data is provided, use a default empty dict
    if students_data is None:
        students_data = {}
        try:
            # Try to import students_data from main app if available
            from assesment5 import students_data
        except ImportError:
            pass
    
    # Create sidebar for AI Tutor navigation
    tutor_view = st.sidebar.radio(
        "AI Tutor View",
        ["Main Tutor", "Learning Resources", "Learning Plan Generator", "Learning Analytics"],
        key="tutor_view_selector"
    )
    
    # Import the AI Tutor functions
    try:
        # Direct import of the functions 
        from ai_tutor_integration import display_main_tutor, display_learning_resources, display_learning_plan_generator, display_learning_analytics
        
        # Display selected view
        if tutor_view == "Main Tutor":
            display_main_tutor(students_data)
        elif tutor_view == "Learning Resources":
            display_learning_resources(students_data)
        elif tutor_view == "Learning Plan Generator":
            display_learning_plan_generator(students_data)
        elif tutor_view == "Learning Analytics":
            display_learning_analytics(students_data)
    except ImportError as e:
        st.error(f"Could not load AI Tutor integration: {e}")
        st.info("Please make sure the ai_tutor_integration.py file is in the same directory as this file.")
        
        # Fallback display
        st.write("## AI Tutor")
        st.write("The AI Tutor helps students with personalized learning assistance.")
        st.write("Features include:")
        st.write("- Interactive learning conversations")
        st.write("- Curated learning resources")
        st.write("- Personalized learning plans")
        st.write("- Performance analytics")
        st.write("- Customized practice materials")
        
        st.info("⚠️ AI Tutor integration module not found. Contact system administrator for support.")

# Function to track student progress (shared with ai_tutor_integration.py)
def track_student_progress(student_id, topic, activity_type, score=None, duration=None):
    """
    Track student progress for the AI Tutor
    
    Args:
        student_id (str): Student ID
        topic (str): Learning topic
        activity_type (str): Type of activity (tutoring, practice, etc.)
        score (float): Score earned (0-1 scale), optional
        duration (int): Time spent in minutes, optional
    """
    # Initialize learning analytics in session state if it doesn't exist
    if "learning_analytics" not in st.session_state:
        st.session_state.learning_analytics = {
            "history": [],
            "topics": {},
            "activities": {
                "tutoring": 0,
                "practice": 0, 
                "quizzes": 0,
                "flashcards": 0,
                "study_notes": 0,
                "learning_plan": 0,
                "resources": 0
            },
            "total_time": 0
        }
    
    # Create activity record
    timestamp = datetime.now()
    activity = {
        "student_id": student_id,
        "topic": topic,
        "activity_type": activity_type,
        "score": score,
        "duration": duration,
        "timestamp": timestamp
    }
    
    # Add to history
    st.session_state.learning_analytics["history"].append(activity)
    
    # Update topic stats
    if topic not in st.session_state.learning_analytics["topics"]:
        st.session_state.learning_analytics["topics"][topic] = {
            "activities": 0,
            "average_score": 0,
            "total_time": 0,
            "last_activity": timestamp
        }
    
    # Update topic statistics
    topic_stats = st.session_state.learning_analytics["topics"][topic]
    topic_stats["activities"] += 1
    topic_stats["last_activity"] = timestamp
    
    if score is not None:
        # Update average score
        current_avg = topic_stats["average_score"]
        current_count = topic_stats["activities"] - 1  # Subtract 1 because we already incremented
        if current_count == 0:
            topic_stats["average_score"] = score
        else:
            new_avg = (current_avg * current_count + score) / topic_stats["activities"]
            topic_stats["average_score"] = new_avg
    
    if duration is not None:
        # Update time spent
        topic_stats["total_time"] += duration
        st.session_state.learning_analytics["total_time"] += duration
    
    # Update activity counts
    if activity_type in st.session_state.learning_analytics["activities"]:
        st.session_state.learning_analytics["activities"][activity_type] += 1
    else:
        # Add new activity type if not exists
        st.session_state.learning_analytics["activities"][activity_type] = 1
