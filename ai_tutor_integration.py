# ai_tutor_integration.py
#
# This file provides the integration between the AI Tutor module and the main assessment system.

import streamlit as st
import openai
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO

# Function to integrate the AI Tutor with the main application
def integrate_ai_tutor(students_data):
    """
    Main function to integrate AI Tutor with the assessment system.
    This should be called from the main application.
    
    Args:
        students_data (dict): Dictionary containing student information
    """
    # Display AI Tutor interface based on view selection
    tutor_view = st.sidebar.radio(
        "AI Tutor View",
        ["Main Tutor", "Learning Resources", "Learning Plan Generator", "Learning Analytics"],
        key="tutor_view_selector"
    )
    
    if tutor_view == "Main Tutor":
        display_main_tutor(students_data)
    elif tutor_view == "Learning Resources":
        display_learning_resources(students_data)
    elif tutor_view == "Learning Plan Generator":
        display_learning_plan_generator(students_data)
    elif tutor_view == "Learning Analytics":
        display_learning_analytics(students_data)

def display_main_tutor(students_data):
    """Display the main AI Tutor chat interface"""
    # Initialize session states if not yet set
    if "tutor_messages" not in st.session_state:
        st.session_state.tutor_messages = [
            {"role": "assistant", "content": "👋 Hello! I'm your AI Tutor. How can I help you learn today?"}
        ]
    
    # Create layout with sidebar for settings and main area for chat
    col1, col2 = st.columns([2, 3])
    
    # First column for settings and learning profile
    with col1:
        st.subheader("Learning Profile")
        
        # Student selection (reuse existing student data)
        student_id = st.selectbox("Select your Student ID", list(students_data.keys()) if students_data else ["DEMO1", "DEMO2"])
        if students_data:
            st.write(f"Student: {students_data[student_id]['name']}")
        else:
            st.write("Student: Demo User")
        
        # Subject/Topic selection
        topic = st.selectbox(
            "What subject would you like to learn?",
            ["Mathematics", "Computer Science", "Physics", "Chemistry", "Biology", 
             "Machine Learning", "Artificial Intelligence", "Data Science", "Statistics"]
        )
        st.session_state.tutor_topic = topic
        
        # Learning style
        learning_style = st.radio(
            "Select your preferred learning style:",
            ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"]
        )
        st.session_state.learning_style = learning_style
        
        # Difficulty level
        difficulty_level = st.select_slider(
            "Select difficulty level:",
            options=["Beginner", "Intermediate", "Advanced", "Expert"]
        )
        st.session_state.difficulty_level = difficulty_level
        
        # Learning goals
        st.subheader("Learning Goals")
        new_goal = st.text_input("Add a learning goal:")
        if st.button("Add Goal") and new_goal:
            if "learning_goals" not in st.session_state:
                st.session_state.learning_goals = []
            st.session_state.learning_goals.append(new_goal)
            
        # Display current goals
        if "learning_goals" in st.session_state and st.session_state.learning_goals:
            for i, goal in enumerate(st.session_state.learning_goals):
                st.write(f"{i+1}. {goal}")
            
            if st.button("Clear Goals"):
                st.session_state.learning_goals = []
        
        # Session analytics
        st.subheader("Session Analytics")
        
        # Calculate session duration
        if "session_start_time" not in st.session_state:
            st.session_state.session_start_time = datetime.now()
            
        current_time = datetime.now()
        session_duration = current_time - st.session_state.session_start_time
        minutes = session_duration.seconds // 60
        
        # Count messages
        num_user_messages = sum(1 for msg in st.session_state.tutor_messages if msg["role"] == "user")
        
        # Display analytics
        st.info(f"Session Duration: {minutes} minutes")
        st.info(f"Messages Exchanged: {len(st.session_state.tutor_messages)}")
        st.info(f"Questions Asked: {num_user_messages}")
        
        # Option to save/download the session
        if st.button("Save Session"):
            # Create a text summary of the session
            session_text = f"AI Tutor Session - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            if students_data and student_id in students_data:
                session_text += f"Student: {students_data[student_id]['name']}\n"
            else:
                session_text += f"Student: Demo User\n"
            session_text += f"Topic: {topic}\n"
            session_text += f"Learning Style: {learning_style}\n"
            session_text += f"Difficulty Level: {difficulty_level}\n\n"
            
            session_text += "Learning Goals:\n"
            if "learning_goals" in st.session_state:
                for i, goal in enumerate(st.session_state.learning_goals):
                    session_text += f"{i+1}. {goal}\n"
            
            session_text += "\nConversation:\n"
            for msg in st.session_state.tutor_messages:
                prefix = "AI Tutor: " if msg["role"] == "assistant" else "You: "
                session_text += f"{prefix}{msg['content']}\n\n"
            
            # Create a download button for the session
            st.download_button(
                label="📥 Download Session Log",
                data=session_text,
                file_name=f"ai_tutor_session_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )
    
    # Second column for the chat interface
    with col2:
        st.subheader("Chat with Your AI Tutor")
        
        # Chat container with custom styling
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.tutor_messages:
                if message["role"] == "assistant":
                    st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>🤖 AI Tutor:</strong> {message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color: #e1f5fe; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>👤 You:</strong> {message['content']}</div>", unsafe_allow_html=True)
        
        # Input for new messages
        with st.form(key="tutor_form", clear_on_submit=True):
            user_input = st.text_area("Your message:", key="tutor_input", height=100)
            
            # Buttons for quick questions/actions
            cols = st.columns(4)
            with cols[0]:
                explain_btn = st.form_submit_button("Explain Topic")
            with cols[1]:
                example_btn = st.form_submit_button("Give Examples")
            with cols[2]:
                practice_btn = st.form_submit_button("Practice Questions")
            with cols[3]:
                submit_btn = st.form_submit_button("Send Message")
            
            # Handle form submission
            if submit_btn and user_input:
                # Add user message to chat
                st.session_state.tutor_messages.append({"role": "user", "content": user_input})
                
                # Generate AI response based on context and learning preferences
                with st.spinner("Thinking..."):
                    # Prepare prompt with context
                    context = f"You are an AI Tutor helping a student learn {st.session_state.tutor_topic}. "
                    context += f"The student's learning style is {st.session_state.learning_style} and they are at a {st.session_state.difficulty_level} level. "
                    
                    if "learning_goals" in st.session_state and st.session_state.learning_goals:
                        context += f"Their learning goals are: {', '.join(st.session_state.learning_goals)}. "
                    
                    messages = [
                        {"role": "system", "content": context},
                    ]
                    
                    # Add conversation history
                    for msg in st.session_state.tutor_messages:
                        messages.append(msg)
                    
                    # Get response from OpenAI
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=messages,
                            temperature=0.7,
                            max_tokens=500
                        )
                        
                        ai_response = response.choices[0].message.content.strip()
                        
                        # Add AI response to chat
                        st.session_state.tutor_messages.append({"role": "assistant", "content": ai_response})
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.session_state.tutor_messages.append({"role": "assistant", "content": "I'm sorry, I encountered an error. Please try again."})
                
                # Track tutoring activity
                track_student_progress(
                    student_id=student_id,
                    topic=st.session_state.tutor_topic,
                    activity_type="tutoring",
                    duration=5  # Estimated time in minutes
                )
                
                # Rerun to update the chat display
                st.experimental_rerun()
            
            # Handle quick action buttons
            elif explain_btn:
                prompt = f"Please explain the core concepts of {st.session_state.tutor_topic} in a way that suits my {st.session_state.learning_style} learning style. Keep it at a {st.session_state.difficulty_level} level."
                st.session_state.tutor_messages.append({"role": "user", "content": prompt})
                st.experimental_rerun()
                
            elif example_btn:
                prompt = f"Could you give me some examples related to {st.session_state.tutor_topic} that would help a {st.session_state.learning_style} learner? Make them appropriate for {st.session_state.difficulty_level} level."
                st.session_state.tutor_messages.append({"role": "user", "content": prompt})
                st.experimental_rerun()
                
            elif practice_btn:
                prompt = f"Please generate some practice questions about {st.session_state.tutor_topic} at a {st.session_state.difficulty_level} difficulty level."
                st.session_state.tutor_messages.append({"role": "user", "content": prompt})
                st.experimental_rerun()
        
        # Additional tools and resources
        st.subheader("Learning Tools")
        tool_cols = st.columns(3)
        
        with tool_cols[0]:
            if st.button("Generate Study Notes"):
                with st.spinner("Generating study notes..."):
                    # Create prompt for study notes
                    notes_prompt = f"Create concise study notes on {st.session_state.tutor_topic} for a {st.session_state.difficulty_level} level student with {st.session_state.learning_style} learning style. Format them in markdown with clear headings, bullet points, and emphasis on key concepts."
                    
                    try:
                        notes_response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are an expert tutor creating study notes."},
                                {"role": "user", "content": notes_prompt}
                            ],
                            temperature=0.7,
                            max_tokens=1000
                        )
                        
                        study_notes = notes_response.choices[0].message.content.strip()
                        
                        # Display and offer download of study notes
                        st.markdown("### Your Study Notes")
                        st.markdown(study_notes)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Study Notes",
                            data=study_notes,
                            file_name=f"{st.session_state.tutor_topic.lower().replace(' ', '_')}_study_notes.md",
                            mime="text/markdown"
                        )
                        
                        # Track activity
                        track_student_progress(
                            student_id=student_id,
                            topic=st.session_state.tutor_topic,
                            activity_type="study_notes",
                            duration=10  # Estimated time in minutes
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating study notes: {str(e)}")
        
        with tool_cols[1]:
            if st.button("Create Flashcards"):
                with st.spinner("Creating flashcards..."):
                    # Generate flashcards
                    flashcards_prompt = f"Create 5 flashcards for learning {st.session_state.tutor_topic} at a {st.session_state.difficulty_level} level. Format as a JSON list with 'question' and 'answer' fields."
                    
                    try:
                        flashcards_response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are creating educational flashcards. Return valid JSON only."},
                                {"role": "user", "content": flashcards_prompt}
                            ],
                            temperature=0.7,
                            max_tokens=800
                        )
                        
                        response_text = flashcards_response.choices[0].message.content.strip()
                        
                        # Extract JSON if it's wrapped in code blocks
                        if "```json" in response_text:
                            response_text = response_text.split("```json")[1].split("```")[0].strip()
                        elif "```" in response_text:
                            response_text = response_text.split("```")[1].split("```")[0].strip()
                            
                        flashcards = json.loads(response_text)
                        
                        # Display flashcards
                        st.markdown("### Flashcards")
                        
                        for i, card in enumerate(flashcards):
                            with st.expander(f"Card {i+1}: {card['question']}"):
                                st.markdown(f"**Answer:** {card['answer']}")
                        
                        # Track activity
                        track_student_progress(
                            student_id=student_id,
                            topic=st.session_state.tutor_topic,
                            activity_type="flashcards",
                            duration=8  # Estimated time in minutes
                        )
                                
                    except Exception as e:
                        st.error(f"Error creating flashcards: {str(e)}")
        
        with tool_cols[2]:
            if st.button("Generate Quiz"):
                with st.spinner("Creating quiz..."):
                    quiz_prompt = f"Create a quick 5-question multiple-choice quiz about {st.session_state.tutor_topic} at a {st.session_state.difficulty_level} level. Format as JSON with 'question', 'options' (array), and 'correct_answer' (index) fields."
                    
                    try:
                        quiz_response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are creating an educational quiz. Return valid JSON only."},
                                {"role": "user", "content": quiz_prompt}
                            ],
                            temperature=0.7,
                            max_tokens=800
                        )
                        
                        response_text = quiz_response.choices[0].message.content.strip()
                        
                        # Extract JSON if it's wrapped in code blocks
                        if "```json" in response_text:
                            response_text = response_text.split("```json")[1].split("```")[0].strip()
                        elif "```" in response_text:
                            response_text = response_text.split("```")[1].split("```")[0].strip()
                            
                        quiz_questions = json.loads(response_text)
                        
                        # Set up quiz in session state
                        st.session_state.ai_tutor_quiz = quiz_questions
                        st.session_state.ai_tutor_quiz_score = 0
                        st.session_state.ai_tutor_current_question = 0
                        st.session_state.ai_tutor_quiz_started = True
                        
                        # Track activity
                        track_student_progress(
                            student_id=student_id,
                            topic=st.session_state.tutor_topic,
                            activity_type="quizzes",
                            duration=15  # Estimated time in minutes
                        )
                        
                        # Refresh the page to show the quiz
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error generating quiz: {str(e)}")
        
        # Display quiz if it's started
        if "ai_tutor_quiz_started" in st.session_state and st.session_state.ai_tutor_quiz_started:
            st.markdown("---")
            st.subheader("Quick Quiz")
            
            if "ai_tutor_quiz" in st.session_state and "ai_tutor_current_question" in st.session_state:
                quiz = st.session_state.ai_tutor_quiz
                current_q_idx = st.session_state.ai_tutor_current_question
                
                if quiz and current_q_idx < len(quiz):
                    # Display current question
                    q = quiz[current_q_idx]
                    st.markdown(f"**Question {current_q_idx+1} of {len(quiz)}:** {q['question']}")
                    
                    # Display options
                    selected_option = st.radio("Choose your answer:", q["options"], key=f"quiz_q{current_q_idx}")
                    selected_idx = q["options"].index(selected_option)
                    
                    if st.button("Submit Answer"):
                        # Check if answer is correct
                        if selected_idx == q["correct_answer"]:
                            st.success("✅ Correct!")
                            st.session_state.ai_tutor_quiz_score += 1
                        else:
                            st.error("❌ Incorrect!")
                            st.info(f"The correct answer was: {q['options'][q['correct_answer']]}")
                        
                        # Move to next question
                        st.session_state.ai_tutor_current_question += 1
                        st.experimental_rerun()
                else:
                    # Quiz completed, show results
                    score = st.session_state.ai_tutor_quiz_score
                    total = len(quiz)
                    st.success(f"Quiz completed! Your score: {score}/{total}")
                    
                    # Progress bar
                    progress = score / total
                    st.progress(progress)
                    
                    # Feedback based on score
                    if progress >= 0.8:
                        st.balloons()
                        st.markdown("**Excellent work!** You have a strong understanding of this topic.")
                    elif progress >= 0.6:
                        st.markdown("**Good job!** You have a solid grasp of the basics.")
                    else:
                        st.markdown("**Keep practicing!** Consider reviewing this topic more thoroughly.")
                    
                    if st.button("Start New Quiz"):
                        # Reset quiz state
                        st.session_state.ai_tutor_quiz_started = False
                        st.experimental_rerun()

def display_learning_resources(students_data):
    """Display curated learning resources for the selected topic"""
    st.title("📚 Learning Resources")
    
    # Initialize session state variables for resources
    if "resource_topic" not in st.session_state:
        st.session_state.resource_topic = st.session_state.get("tutor_topic", "Computer Science")
    
    if "resource_level" not in st.session_state:
        st.session_state.resource_level = st.session_state.get("difficulty_level", "Intermediate")
    
    if "saved_resources" not in st.session_state:
        st.session_state.saved_resources = []
    
    # Set up sidebar for filtering resources
    st.sidebar.subheader("Resource Filters")
    
    # Topic selection
    resource_topic = st.sidebar.selectbox(
        "Topic",
        ["Computer Science", "Mathematics", "Physics", "Chemistry", "Biology", 
         "Machine Learning", "Artificial Intelligence", "Data Science", "Statistics",
         "Web Development", "Mobile Development", "Game Development", "Cybersecurity"],
        index=["Computer Science", "Mathematics", "Physics", "Chemistry", "Biology", 
               "Machine Learning", "Artificial Intelligence", "Data Science", "Statistics",
               "Web Development", "Mobile Development", "Game Development", "Cybersecurity"].index(
               st.session_state.resource_topic) if st.session_state.resource_topic in ["Computer Science", "Mathematics", "Physics", "Chemistry", "Biology", 
               "Machine Learning", "Artificial Intelligence", "Data Science", "Statistics",
               "Web Development", "Mobile Development", "Game Development", "Cybersecurity"] else 0
    )
    st.session_state.resource_topic = resource_topic
    
    # Specify a subtopic for more targeted resources
    subtopic = st.sidebar.text_input("Specific Subject/Subtopic (optional)", 
                                    help="For more specific resources, enter a subtopic (e.g., 'Python', 'Linear Algebra', 'Neural Networks')")
    
    # Resource difficulty level
    resource_level = st.sidebar.select_slider(
        "Difficulty Level",
        options=["Beginner", "Intermediate", "Advanced", "Expert"],
        value=st.session_state.resource_level
    )
    st.session_state.resource_level = resource_level
    
    # Resource type filter
    resource_types = st.sidebar.multiselect(
        "Resource Types",
        ["Books", "Online Courses", "Video Tutorials", "Interactive Tutorials", "Documentation", 
         "Academic Papers", "Blogs/Articles", "Podcasts", "GitHub Repositories", "Practice Problems"],
        default=["Books", "Online Courses", "Video Tutorials"]
    )
    
    # Free vs paid resources
    cost_filter = st.sidebar.radio("Cost", ["All Resources", "Free Only", "Paid Only"], index=0)
    
    # Generate resources button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Learning Resources for {subtopic + ' ' if subtopic else ''}{resource_topic}")
    with col2:
        generate_btn = st.button("🔄 Regenerate Resources", use_container_width=True)
    
    # Student selection for analytics tracking
    student_id = st.selectbox("Select Student", list(students_data.keys()) if students_data else ["DEMO1", "DEMO2"], key="resource_student_id")
    if students_data and student_id in students_data:
        student_name = students_data[student_id]["name"]
    else:
        student_name = "Demo User"
    
    # Function to generate and display resources
    def generate_resources():
        with st.spinner("Generating curated learning resources..."):
            # Prepare prompt for resource generation
            resource_prompt = f"""
            Generate a comprehensive list of learning resources for {subtopic + ' in ' if subtopic else ''}{resource_topic} at a {resource_level} level.
            
            Include the following types of resources:
            {', '.join(resource_types)}
            
            {f'Only include free resources.' if cost_filter == 'Free Only' else ''}
            {f'Only include paid/premium resources.' if cost_filter == 'Paid Only' else ''}
            
            For each resource, provide:
            1. Title
            2. Type (book, course, video, etc.)
            3. Author/Creator/Platform
            4. Brief description (1-2 sentences)
            5. Difficulty level
            6. Cost (free, paid, freemium, etc.)
            7. URL if applicable
            8. Why it's valuable for this topic
            
            Format the response in JSON as an array of resource objects with the fields above.
            """
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an AI specialized in curating educational resources. Return valid JSON only."},
                        {"role": "user", "content": resource_prompt}
                    ],
                    temperature=0.5,
                    max_tokens=1500
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Extract JSON if it's wrapped in code blocks
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                resources = json.loads(response_text)
                
                # Track resource lookup activity
                track_student_progress(
                    student_id=student_id,
                    topic=resource_topic,
                    activity_type="resources",
                    duration=5  # Estimated time in minutes
                )
                
                return resources
                
            except Exception as e:
                st.error(f"Error generating resources: {str(e)}")
                st.write("Response received:", response_text if 'response_text' in locals() else "No response")
                return None
    
    # Check if we need to generate new resources
    resources_key = f"{resource_topic}_{subtopic}_{resource_level}_{'-'.join(resource_types)}_{cost_filter}"
    if "current_resources" not in st.session_state or "current_resources_key" not in st.session_state or st.session_state.current_resources_key != resources_key or generate_btn:
        resources = generate_resources()
        if resources:
            st.session_state.current_resources = resources
            st.session_state.current_resources_key = resources_key
    
    # Display resources
    if "current_resources" in st.session_state and st.session_state.current_resources:
        resources = st.session_state.current_resources
        
        # Group resources by type
        resource_by_type = {}
        for resource in resources:
            resource_type = resource.get("type", "Other")
            if resource_type not in resource_by_type:
                resource_by_type[resource_type] = []
            resource_by_type[resource_type].append(resource)
        
        # Display resources by type
        for resource_type, type_resources in resource_by_type.items():
            with st.expander(f"{resource_type.title()} ({len(type_resources)})", expanded=True):
                for i, resource in enumerate(type_resources):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"### {resource.get('title', 'Untitled Resource')}")
                        st.markdown(f"**Author/Creator:** {resource.get('author', 'Unknown')}")
                        st.markdown(f"**Description:** {resource.get('description', 'No description available')}")
                        st.markdown(f"**Difficulty:** {resource.get('difficulty', 'Not specified')} | **Cost:** {resource.get('cost', 'Not specified')}")
                        
                        if "why_valuable" in resource:
                            st.markdown(f"**Why it's valuable:** {resource.get('why_valuable')}")
                            
                        if "url" in resource and resource["url"]:
                            st.markdown(f"**Link:** [{resource['url']}]({resource['url']})")
                            
                    with col2:
                        # Save resource button
                        if st.button("Save", key=f"save_{resource_type}_{i}"):
                            if resource not in st.session_state.saved_resources:
                                st.session_state.saved_resources.append(resource)
                                st.success("Resource saved!")
                                st.experimental_rerun()
                    
                    st.markdown("---")
        
        # Create a markdown document with all resources
        if st.button("📥 Download All Resources"):
            markdown_content = f"# Learning Resources for {subtopic + ' in ' if subtopic else ''}{resource_topic}\n\n"
            markdown_content += f"**Difficulty Level:** {resource_level}\n\n"
            
            for resource_type, type_resources in resource_by_type.items():
                markdown_content += f"## {resource_type.title()}\n\n"
                
                for resource in type_resources:
                    markdown_content += f"### {resource.get('title', 'Untitled Resource')}\n"
                    markdown_content += f"**Author/Creator:** {resource.get('author', 'Unknown')}  \n"
                    markdown_content += f"**Description:** {resource.get('description', 'No description available')}  \n"
                    markdown_content += f"**Difficulty:** {resource.get('difficulty', 'Not specified')} | **Cost:** {resource.get('cost', 'Not specified')}  \n"
                    
                    if "why_valuable" in resource:
                        markdown_content += f"**Why it's valuable:** {resource.get('why_valuable')}  \n"
                        
                    if "url" in resource and resource["url"]:
                        markdown_content += f"**Link:** [{resource['url']}]({resource['url']})  \n"
                    
                    markdown_content += "\n---\n\n"
            
            st.download_button(
                label="📥 Download as Markdown",
                data=markdown_content,
                file_name=f"{resource_topic.lower().replace(' ', '_')}_resources.md",
                mime="text/markdown"
            )
    
    # Display saved resources
    if st.session_state.saved_resources:
        st.sidebar.subheader("My Saved Resources")
        for i, resource in enumerate(st.session_state.saved_resources):
            with st.sidebar.expander(f"{resource.get('title', 'Untitled Resource')}", expanded=False):
                st.markdown(f"**Type:** {resource.get('type', 'Unknown')}")
                st.markdown(f"**Author:** {resource.get('author', 'Unknown')}")
                
                if "url" in resource and resource["url"]:
                    st.markdown(f"[Open Resource]({resource['url']})")
                
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.saved_resources.pop(i)
                    st.experimental_rerun()
        
        if st.sidebar.button("Clear All Saved Resources"):
            st.session_state.saved_resources = []
            st.experimental_rerun()

def display_learning_plan_generator(students_data):
    """Display interface for generating personalized learning plans"""
    st.title("📝 AI Tutor - Personalized Learning Plan Generator")
    
    # Student selection
    student_id = st.selectbox("Select Student", list(students_data.keys()) if students_data else ["DEMO1", "DEMO2"])
    if students_data and student_id in students_data:
        student_name = students_data[student_id]["name"]
    else:
        student_name = "Demo User"
    st.write(f"Generating learning plan for: **{student_name}**")
    
    # Plan parameters
    col1, col2 = st.columns(2)
    
    with col1:
        if "tutor_topic" not in st.session_state:
            st.session_state.tutor_topic = "Computer Science"
            
        if "learning_style" not in st.session_state:
            st.session_state.learning_style = "Visual"
            
        if "difficulty_level" not in st.session_state:
            st.session_state.difficulty_level = "Intermediate"
            
        topic = st.text_input("Learning Topic", value=st.session_state.tutor_topic)
        learning_style = st.radio("Learning Style", 
                                 ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"],
                                 index=["Visual", "Auditory", "Reading/Writing", "Kinesthetic"].index(st.session_state.learning_style))
    
    with col2:
        difficulty = st.select_slider("Difficulty Level", 
                                     options=["Beginner", "Intermediate", "Advanced", "Expert"],
                                     value=st.session_state.difficulty_level)
        duration = st.slider("Plan Duration (weeks)", min_value=1, max_value=12, value=4)
    
    # Learning goals
    st.subheader("Learning Goals")
    st.caption("Add specific learning goals to customize your plan")
    
    if "learning_plan_goals" not in st.session_state:
        st.session_state.learning_plan_goals = []
    
    # Add goals
    new_goal = st.text_input("Add a learning goal:")
    if st.button("Add Goal") and new_goal:
        st.session_state.learning_plan_goals.append(new_goal)
    
    # Display and allow removal of goals
    if st.session_state.learning_plan_goals:
        goals_to_remove = []
        for i, goal in enumerate(st.session_state.learning_plan_goals):
            col1, col2 = st.columns([10, 1])
            with col1:
                st.write(f"{i+1}. {goal}")
            with col2:
                if st.button("🗑️", key=f"remove_goal_{i}"):
                    goals_to_remove.append(i)
        
        # Remove goals that were marked for deletion
        for i in sorted(goals_to_remove, reverse=True):
            st.session_state.learning_plan_goals.pop(i)
        
        if st.button("Clear All Goals"):
            st.session_state.learning_plan_goals = []
            st.experimental_rerun()
    
    # Generate plan button
    if st.button("Generate Learning Plan", type="primary"):
        with st.spinner("Creating your personalized learning plan..."):
            # Update prompt with goals if provided
            goals_text = ""
            if st.session_state.learning_plan_goals:
                goals_text = "The student has the following specific learning goals:\n"
                for goal in st.session_state.learning_plan_goals:
                    goals_text += f"- {goal}\n"
            
            # Generate the plan
            prompt = f"""
            Create a {duration}-week personalized learning plan for a student studying {topic} at a {difficulty} level.
            The student has a {learning_style} learning style preference.
            
            {goals_text}
            
            Format the plan as follows:
            1. Include a brief overview of what will be covered
            2. Divide the plan into {duration} weeks
            3. For each week, include:
               - Learning objectives for the week
               - Recommended learning activities (readings, exercises, projects, etc.)
               - Estimated time commitment per day
               - Assessment/checkpoint to verify understanding
            4. Tailor activities to suit a {learning_style} learner
            
            Return the learning plan as markdown. Keep it concise but comprehensive.
            """
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are creating a personalized educational learning plan."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=1500
                )
                
                learning_plan = response.choices[0].message.content.strip()
                
                # Add header with student info
                header = f"""# Personalized Learning Plan
**Student:** {student_name} (ID: {student_id})  
**Topic:** {topic}  
**Difficulty Level:** {difficulty}  
**Learning Style:** {learning_style}  
**Duration:** {duration} weeks  
**Generated:** {datetime.now().strftime('%Y-%m-%d')}

---

"""
                
                full_plan = header + learning_plan
                
                if full_plan:
                    st.session_state.current_learning_plan = full_plan
                    st.success("Learning plan generated successfully!")
                    
                    # Display the plan
                    st.markdown("### Your Personalized Learning Plan")
                    st.markdown(full_plan)
                    
                    # Offer download
                    st.download_button(
                        label="📥 Download Learning Plan",
                        data=full_plan,
                        file_name=f"{student_name.lower().replace(' ', '_')}_{topic.lower().replace(' ', '_')}_learning_plan.md",
                        mime="text/markdown"
                    )
                    
                    # Track the activity
                    track_student_progress(
                        student_id=student_id,
                        topic=topic,
                        activity_type="learning_plan",
                        duration=5  # Estimated time to create plan (minutes)
                    )
                else:
                    st.error(f"Failed to generate learning plan.")
            except Exception as e:
                st.error(f"Failed to generate learning plan: {str(e)}")
    
    # Display previous plans if available
    if "current_learning_plan" in st.session_state and st.session_state.current_learning_plan:
        st.markdown("---")
        with st.expander("View Current Learning Plan"):
            st.markdown(st.session_state.current_learning_plan)

def display_learning_analytics(students_data):
    """Display learning analytics and progress tracking for students"""
    st.title("📊 AI Tutor - Learning Analytics")
    
    # Ensure we have analytics data
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
    
    # Student selection
    student_id = st.selectbox("Select Student", list(students_data.keys()) if students_data else ["DEMO1", "DEMO2"])
    if students_data and student_id in students_data:
        student_name = students_data[student_id]["name"]
    else:
        student_name = "Demo User"
    
    # Generate some dummy data if we don't have enough
    if len(st.session_state.learning_analytics["history"]) < 3:
        generate_sample_analytics_data(student_id, student_name)
    
    # Analytics container
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Activity Distribution")
        # Create pie chart of activities
        activities = st.session_state.learning_analytics["activities"]
        activity_labels = list(activities.keys())
        activity_values = list(activities.values())
        
        if sum(activity_values) > 0:  # Only show if we have data
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(activity_values, labels=activity_labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            st.pyplot(fig)
        else:
            st.info("No activity data available yet. Start using the AI Tutor to generate analytics.")
    
    with col2:
        st.subheader("Topic Mastery")
        # Create bar chart for topic mastery
        topic_names = ["Variables", "Functions", "Loops", "Classes", "Algorithms"]
        topic_scores = [0.8, 0.6, 0.7, 0.4, 0.3]  # Sample scores
        
        fig, ax = plt.subplots(figsize=(5, 5))
        bars = ax.barh(topic_names, topic_scores, color='skyblue')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Mastery Level')
        
        # Add percentage labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width*100:.0f}%', va='center')
        
        st.pyplot(fig)
    
    # Recent activity timeline
    st.subheader("Recent Learning Activities")
    
    # Filter for only this student's activities
    student_history = [
        activity for activity in st.session_state.learning_analytics["history"] 
        if activity["student_id"] == student_id
    ]
    
    if student_history:
        # Sort by timestamp, most recent first
        student_history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Display in a table
        activity_data = []
        for activity in student_history[:10]:  # Show last 10 activities
            activity_data.append({
                "Date": activity["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "Activity Type": activity["activity_type"].title(),
                "Topic": activity["topic"],
                "Duration (min)": activity["duration"] if activity["duration"] else "-",
                "Score": f"{activity['score']*100:.1f}%" if activity["score"] is not None else "-"
            })
        
        activity_df = pd.DataFrame(activity_data)
        st.dataframe(activity_df)
    else:
        st.info("No activity history available yet. Start using the AI Tutor to generate analytics.")
    
    # Learning time analysis
    st.subheader("Learning Time Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Weekly time spent
        week_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekly_hours = [2.5, 1.0, 3.0, 2.0, 1.5, 4.0, 3.5]  # Sample data
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(week_days, weekly_hours, color='lightgreen')
        ax.set_ylabel('Hours')
        ax.set_title('Weekly Study Pattern')
        st.pyplot(fig)
    
    with col2:
        # Time by activity type
        activity_types = list(st.session_state.learning_analytics["activities"].keys())
        activity_times = [10, 45, 30, 15, 25, 20, 15]  # Sample data - time in minutes
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(activity_types, activity_times, color='lightblue')
        ax.set_xlabel('Minutes')
        ax.set_title('Time by Activity Type')
        st.pyplot(fig)
    
    # Recommendations based on analytics
    st.subheader("AI Tutor Recommendations")
    
    recommendations_prompt = f"""
    Based on the following learning analytics for {student_name},
    provide 3 personalized recommendations to improve their learning experience:
    
    - Most studied topics: Computer Science, Machine Learning
    - Learning style: Visual
    - Average session duration: 25 minutes
    - Activity distribution: Quizzes (40%), Tutoring (30%), Flashcards (20%), Study Notes (10%)
    - Areas for improvement: Algorithms (30% mastery), Classes (40% mastery)
    
    Format as bullet points. Keep each recommendation concise but specific.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI providing educational analytics and recommendations."},
                {"role": "user", "content": recommendations_prompt}
            ],
            temperature=0.5,
            max_tokens=250
        )
        
        recommendations = response.choices[0].message.content.strip()
        
        # Display recommendations in a nice box
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;">
        <h4>Personalized Recommendations</h4>
        {recommendations}
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        st.info("• Increase practice time with Algorithms and Classes to improve mastery\n• Try more visual learning materials like diagrams and videos\n• Break study sessions into shorter, more frequent intervals")

def generate_sample_analytics_data(student_id, student_name):
    """Generate sample analytics data for demonstration purposes"""
    if "learning_analytics" not in st.session_state:
        # Initialize with a basic structure
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
    
    # Sample topics
    topics = ["Computer Science", "Machine Learning", "Data Structures", "Algorithms", "Python Programming"]
    
    # Sample activities
    activity_types = ["tutoring", "practice", "quizzes", "flashcards", "study_notes", "resources"]
    
    # Generate random history entries
    for i in range(10):
        # Random date within the last 30 days
        days_ago = np.random.randint(0, 30)
        timestamp = datetime.now() - pd.Timedelta(days=days_ago)
        
        # Random topic and activity
        topic = np.random.choice(topics)
        activity_type = np.random.choice(activity_types)
        
        # Create activity record
        activity = {
            "student_id": student_id,
            "topic": topic,
            "activity_type": activity_type,
            "score": np.random.random() if activity_type == "quizzes" else None,
            "duration": np.random.randint(5, 60),  # 5-60 minutes
            "timestamp": timestamp
        }
        
        # Add to history
        st.session_state.learning_analytics["history"].append(activity)
        
        # Update activity counts
        st.session_state.learning_analytics["activities"][activity_type] += 1
        
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
        
        if activity["score"] is not None:
            # Update average score
            current_avg = topic_stats["average_score"]
            current_count = topic_stats["activities"] - 1  # Subtract 1 because we already incremented
            if current_count == 0:
                topic_stats["average_score"] = activity["score"]
            else:
                new_avg = (current_avg * current_count + activity["score"]) / topic_stats["activities"]
                topic_stats["average_score"] = new_avg
        
        if activity["duration"] is not None:
            # Update time spent
            topic_stats["total_time"] += activity["duration"]
            st.session_state.learning_analytics["total_time"] += activity["duration"]

# Function to track student progress
def track_student_progress(student_id, topic, activity_type, score=None, duration=None):
    """
    Track student progress for analytics
    
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
