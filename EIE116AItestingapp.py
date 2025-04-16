import pandas as pd
import os
import json
from io import StringIO
import datetime
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager


# -----------------------------------------------------------------------------
# 1) PAGE CONFIG & LOGO
# -----------------------------------------------------------------------------
st.set_page_config(page_title="EIE116 AI Open Book Unproctored Quiz SASTRA", layout="centered")
st.image("sastra_logo.jpg", use_container_width=True)

# -----------------------------------------------------------------------------
# 2) ADMIN PASSWORD
# -----------------------------------------------------------------------------
ADMIN_PASSWORD = "venkat123"
STUDENT_PASSWORD = "ai2025"  # <-- Add a universal student password here

# -----------------------------------------------------------------------------
# 3) STUDENTS DICTIONARY (66 entries)
# -----------------------------------------------------------------------------
STUDENTS = {
    "C2321": "Venkatesh T",
    "127179001": "Aakash M B",
    "127179002": "Abinaya V",
    "127179003": "Achanta Sasi Kumar",
    "127179004": "Adhithya Vishwa A",
    "127179005": "Ananya Sridhar",
    "127179007": "Avinandan Pal",
    "127179008": "Buvaneshwaran M",
    "127179009": "Kesarla Chaitrika",
    "127179010": "Charulatha B",
    "127179011": "Dravid Abishek M",
    "127179012": "Gajula Devakumargari Varshitha",
    "127179013": "Glenn Mcgrath A",
    "127179014": "Gokulessh S V",
    "127179015": "Govinda Roobini J D",
    "127179016": "H R Krishnavamsi",
    "127179017": "Hari Sathish S",
    "127179018": "Harini Mahesh Kumar",
    "127179019": "Harsha D",
    "127179020": "Imam Imthiyaz Basha I",
    "127179021": "Jaisnav S P",
    "127179022": "Janakeerthana B",
    "127179023": "Jeeva Shamishta V",
    "127179024": "Kalayigiri Rabia",
    "127179025": "Kammara Nihaleswar",
    "127179026": "Kavinraj S B",
    "127179027": "Keerthiga M",
    "127179028": "Kelwin George",
    "127179029": "Kishore M",
    "127179030": "Kothagundu Krishna Saatvik",
    "127179031": "Kulukuri Bala Surya Sathwik",
    "127179032": "Leenaa S",
    "127179033": "Madhapur Raghasree",
    "127179034": "Bhargav Sreenivas N",
    "127179035": "Nalavala Venkata Kalyan Reddy",
    "127179036": "Nalla Perumal Aayush C",
    "127179037": "Nanda Krishnan N",
    "127179038": "Oleti Lasya Sri",
    "127179039": "Ommi Reshma",
    "127179040": "Palli Dhanush Eswar",
    "127179041": "Penjarla Veera Venkata Sai Sriramsathvik",
    "127179042": "Pranav R Raaj",
    "127179043": "Prashanth Gandhi",
    "127179044": "Pujari Meghana",
    "127179045": "Ragul N",
    "127179046": "Rayidi Neeharika",
    "127179047": "Ryali Sreeshanth",
    "127179048": "Mohita Varshini S",
    "127179049": "Sabarish C",
    "127179050": "Sakthi S",
    "127179051": "Sarabu Radha Krishna Santhosh",
    "127179052": "Senthamilselvan R",
    "127179053": "Shabari M",
    "127179055": "Shriram Krishna T",
    "127179056": "Vasantham Sowmya",
    "127179057": "Sreya S",
    "127179058": "Sushanth Suresh",
    "127179059": "Sushmitha Rajmohan",
    "127179060": "Talanki Mamatha",
    "127179062": "Thejaswini A S",
    "127179063": "Uma Parvathi G",
    "127179064": "Varshini S",
    "127179065": "Vattam Amrutha",
    "127179066": "Vedula Uday Easwar",
    "127179067": "Vidhya Lakshmi R",
    "127179068": "Vishnuvikas U",
    "127179069": "Yattapu Jahnavi"
}

# -----------------------------------------------------------------------------
# 4) QUESTIONS
# -----------------------------------------------------------------------------
QUESTIONS = [
    {
        "question": "Q1.1: You have a 5×5 grid with rewards, obstacles, or free cells. Why can DFS still find a “most rewarding path”?",
        "options": [
            "A. DFS always explores all possible paths and can compare total rewards.",
            "B. DFS always prefers the path with highest immediate reward at each step.",
            "C. DFS guarantees the shortest path in distance.",
            "D. DFS uses a priority queue based on heuristic values."
        ],
        "answer": "A"
    },
    {
        "question": "Q1.2: Which technique helps DFS skip paths that cannot exceed the best reward so far?",
        "options": [
            "A. Using a heuristic function that estimates distance.",
            "B. Pruning paths that can’t beat the current best total.",
            "C. Running BFS first for path length, then discarding longer ones.",
            "D. Turning DFS into uniform cost search."
        ],
        "answer": "B"
    },
    {
        "question": "Q2.1: BFS in a 5×5 grid with danger levels ensures the path is safest by...",
        "options": [
            "A. Expanding paths with the lowest cumulative danger.",
            "B. Exploring neighbors level by level, can adapt to treat high danger as blocked.",
            "C. Guaranteeing the path with the minimum steps is also the safest path.",
            "D. Being unable to adapt BFS for danger levels."
        ],
        "answer": "C"
    },
    {
        "question": "Q2.2: If two equally short paths exist but different dangers, standard BFS will...",
        "options": [
            "A. Pick the first path it finds (both are same distance).",
            "B. Always choose the path with the lowest total danger.",
            "C. Alternate expansions to compare them fairly.",
            "D. Fail to find a path if multiple equally short paths exist."
        ],
        "answer": "A"
    },
    {
        "question": "Q3.1: Best First Search with danger levels prioritizes the next cell by...",
        "options": [
            "A. The lowest danger value (heuristic).",
            "B. The highest danger value (heuristic).",
            "C. The sum of path cost plus danger estimate.",
            "D. Random selection of the next cell."
        ],
        "answer": "A"
    },
    {
        "question": "Q3.2: A common pitfall of Best First Search with danger levels is...",
        "options": [
            "A. It can get stuck in loops ignoring danger heuristics.",
            "B. It may choose a very long path if the heuristic is purely danger-based.",
            "C. It must explore all paths, so it’s slower than DFS.",
            "D. It behaves exactly like BFS."
        ],
        "answer": "B"
    },
    {
        "question": "Q4.1: A* in a 5×5 grid with variable costs finds the path by prioritizing the node with...",
        "options": [
            "A. Lowest g(n).",
            "B. Lowest h(n).",
            "C. Lowest g(n) + h(n).",
            "D. Smallest edge to a neighbor."
        ],
        "answer": "C"
    },
    {
        "question": "Q4.2: Manhattan distance is often used as a heuristic in A* because...",
        "options": [
            "A. It only applies if diagonal moves are allowed.",
            "B. It’s admissible for 4-direction moves in a uniform grid.",
            "C. It always overestimates the true cost.",
            "D. It can’t be used in real road networks."
        ],
        "answer": "B"
    },
    {
        "question": "Q5.1: For a 5×5 grid with different costs, Dijkstra’s algorithm...",
        "options": [
            "A. Only works if costs are all 1.",
            "B. Finds the path minimizing total cost for nonnegative edges.",
            "C. Always requires a heuristic.",
            "D. Is the same as DFS."
        ],
        "answer": "B"
    },
    {
        "question": "Q5.2: In Dijkstra’s algorithm, we typically use which data structure to pick the smallest path cost node?",
        "options": [
            "A. A list we scan each time.",
            "B. A stack (LIFO).",
            "C. A max-heap.",
            "D. A min-heap priority queue."
        ],
        "answer": "D"
    },
    {
        "question": "Q6.1: Backtracking in a 5×5 grid with obstacles/danger means we...",
        "options": [
            "A. Use a heuristic to expand nodes with smallest cost + danger.",
            "B. Explore every path, backtrack on obstacles, store a path if ‘D’ is reached.",
            "C. Only expand safe nodes if they have lower danger than previous ones.",
            "D. Use a queue to manage expansions level by level."
        ],
        "answer": "B"
    },
    {
        "question": "Q6.2: In backtracking for a 5×5 grid, we avoid revisiting the same cell in the same path by...",
        "options": [
            "A. Marking each visited cell to not revisit within that path recursion.",
            "B. Relying on BFS alone.",
            "C. Cannot avoid re-visits; backtracking tries all possibilities.",
            "D. Using a priority queue to filter repeats."
        ],
        "answer": "A"
    },
    {
        "question": "Q7.1: In DFS with pruning for the shortest path from (4,2) to (0,4), a typical prune is...",
        "options": [
            "A. If current path cost > best known, backtrack.",
            "B. If the node danger is higher than previously visited nodes, skip it.",
            "C. If the node was visited, always skip it (even if that might yield a better path).",
            "D. We never prune in DFS."
        ],
        "answer": "A"
    },
    {
        "question": "Q7.2: Searching for the shortest path with DFS + pruning, we can prune a path if...",
        "options": [
            "A. We found a shorter route to that node before.",
            "B. We suspect there might be a better path but continue anyway.",
            "C. The node is unvisited, so prune immediately.",
            "D. We never prune in DFS."
        ],
        "answer": "A"
    },
    {
        "question": "Q8.1: Scheduling classes to avoid teacher/room conflicts is usually solved with...",
        "options": [
            "A. Backtracking + constraint checks.",
            "B. Dijkstra’s algorithm.",
            "C. BFS by timeslot expansions.",
            "D. Q-learning."
        ],
        "answer": "A"
    },
    {
        "question": "Q8.2: In a scheduling CSP with backtracking, a technique to reduce the search space is...",
        "options": [
            "A. Relying on a random assignment first.",
            "B. Using forward checking to remove impossible assignments early.",
            "C. Always assigning teachers last.",
            "D. Shuffle constraints arbitrarily each time."
        ],
        "answer": "B"
    },
    {
        "question": "Q9.1: For Classes={Math, Physics, English}, Teachers={T1, T2, T3}, Rooms={R1,R2}, Time Slots={9AM,10AM}, we must ensure...",
        "options": [
            "A. Each teacher has exactly one class, ignoring times.",
            "B. No teacher or room is double-booked, and all classes fit in timeslots.",
            "C. We rely on cost functions for picking timeslots automatically.",
            "D. BFS is the only approach."
        ],
        "answer": "B"
    },
    {
        "question": "Q9.2: A systematic way to assign classes/teachers/rooms/time is...",
        "options": [
            "A. BFS enumerating all times first.",
            "B. Backtracking: assign one variable at a time, check constraints.",
            "C. Q-learning to minimize idle time.",
            "D. Repeated DFS ignoring constraints."
        ],
        "answer": "B"
    },
    {
        "question": "Q10.1: Depth-First Search from A to G in a graph...",
        "options": [
            "A. Guarantees the shortest path.",
            "B. Explores one path fully, might not find the shortest, but will find a path if it exists.",
            "C. Uses a heuristic to prioritize expansions.",
            "D. Never visits the same node more than once in any path."
        ],
        "answer": "B"
    },
    {
        "question": "Q10.2: To make DFS from A to G more efficient in a large graph, we...",
        "options": [
            "A. Remove nodes from the graph after visiting them.",
            "B. Convert DFS into BFS.",
            "C. Keep a visited set to avoid revisiting nodes.",
            "D. Double the edges to see all paths."
        ],
        "answer": "C"
    },
    {
        "question": "Q11.1: BFS from A to G in an unweighted graph...",
        "options": [
            "A. Expands neighbors by ascending heuristic value.",
            "B. Expands nodes level by level, guaranteeing the shortest path in edge count.",
            "C. Identical to DFS in approach.",
            "D. Cannot handle repeated states."
        ],
        "answer": "B"
    },
    {
        "question": "Q11.2: To find the shortest path from A to G with BFS, we typically...",
        "options": [
            "A. Expand nodes by ascending heuristic value.",
            "B. Use a queue, enqueue neighbors as discovered, stop when G is found.",
            "C. Pick the path with maximum immediate reward.",
            "D. Use a stack to explore deeply."
        ],
        "answer": "B"
    },
    {
        "question": "Q12.1: Best First Search from S to E in a graph...",
        "options": [
            "A. Expands the node with the lowest h(n).",
            "B. Expands the node with the lowest g(n) + h(n).",
            "C. Expands the node with the lowest g(n).",
            "D. Chooses neighbors randomly."
        ],
        "answer": "A"
    },
    {
        "question": "Q12.2: Best First Search can get stuck in a dead end if...",
        "options": [
            "A. It always picks the path with the largest cost first.",
            "B. It uses f(n)=g(n)+h(n).",
            "C. Its heuristic leads to exploring a node that has no path to E beyond it.",
            "D. It marks all unvisited nodes as blocked."
        ],
        "answer": "C"
    },
    {
        "question": "Q13.1: Main drawback of Best First Search is...",
        "options": [
            "A. It always returns the shortest path.",
            "B. Ignores the distance traveled (g(n)), so may yield suboptimal routes.",
            "C. It never uses heuristics, purely random.",
            "D. Only runs on grids, not graphs."
        ],
        "answer": "B"
    },
    {
        "question": "Q13.2: If we repeat Best First Search multiple times from S to E, we can optimize by...",
        "options": [
            "A. Caching visited states and their best known heuristics.",
            "B. Decreasing the heuristic to near 0.",
            "C. Switching to random expansion.",
            "D. Only exploring zero-cost neighbors."
        ],
        "answer": "A"
    },
    {
        "question": "Q13.3: If Best First Search follows a promising path that fails, it typically...",
        "options": [
            "A. Switches to BFS halfway through.",
            "B. Jumps randomly to unvisited nodes.",
            "C. Uses backtracking to explore alternative branches.",
            "D. Ignores all other possible paths."
        ],
        "answer": "C"
    },
    {
        "question": "Q14.1: For Q-learning in a 4x4 grid, the standard Q-update rule is...",
        "options": [
            "A. Q(s,a) = Q(s,a) + α[r + γ*max Q(s',a') - Q(s,a)].",
            "B. Q(s,a) = r + max Q(s',a').",
            "C. Q(s,a) = α*r, ignoring future rewards.",
            "D. Q(s,a) = 0 at every step."
        ],
        "answer": "A"
    },
    {
        "question": "Q14.2: Exploration vs. exploitation in Q-learning is typically handled by...",
        "options": [
            "A. Always picking the highest Q-value action (0% exploration).",
            "B. An ε-greedy approach: random action with prob. ε, else best known action.",
            "C. Only random actions until all states are visited.",
            "D. Picking the worst action to ensure coverage."
        ],
        "answer": "B"
    },
    {
        "question": "Q14.3: Once Q-learning converges, the optimal policy is recovered by...",
        "options": [
            "A. Selecting the action with the highest Q(s,a) at each state s.",
            "B. Averaging all Q-values in each state.",
            "C. Restarting from scratch.",
            "D. Random moves until a path emerges."
        ],
        "answer": "A"
    },
]

# -----------------------------------------------------------------------------
# 5) CSV FILE FOR PERSISTENCE
# -----------------------------------------------------------------------------
DATA_FILE = "test_data.csv"

def load_data():
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=["student_id", "status", "score", "answers_json", "submitted_at"])
        df.to_csv(DATA_FILE, index=False)
    else:
        df = pd.read_csv(DATA_FILE)
    return df

def save_data(df: pd.DataFrame):
    df.to_csv(DATA_FILE, index=False)

def get_user_record(df: pd.DataFrame, student_id: str):
    row = df.loc[df["student_id"] == student_id]
    if row.empty:
        return None
    else:
        return row.iloc[0].to_dict()

def update_user_record(df: pd.DataFrame, student_id: str, status: str, score: int, answers_list: list):
    answers_json = json.dumps(answers_list)
    existing = df.loc[df["student_id"] == student_id]
    if existing.empty:
        new_row = {
            "student_id": student_id,
            "status": status,
            "score": score,
            "answers_json": answers_json
        }
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)
    else:
        idx = existing.index[0]
        df.at[idx, "status"] = status
        df.at[idx, "score"] = score
        df.at[idx, "answers_json"] = answers_json
    return df

def compute_score(answers_list: list) -> int:
    correct_count = 0
    for i, ans in enumerate(answers_list):
        if ans.startswith(QUESTIONS[i]["answer"]):
            correct_count += 1
    return correct_count

def download_all_results(df: pd.DataFrame) -> str:
    max_q = len(QUESTIONS)
    columns = ["student_id", "status", "score"] + [f"Q{i+1}" for i in range(max_q)]
    rows = []

    for _, row in df.iterrows():
        answers_list = json.loads(row["answers_json"])
        row_out = [
            row["student_id"],
            row["status"],
            f"{row['score']}/{max_q}",
        ]
        for ans in answers_list:
            row_out.append(ans[0] if ans else "")
        rows.append(row_out)

    final_df = pd.DataFrame(rows, columns=columns)
    csv_buffer = StringIO()
    final_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

# -----------------------------------------------------------------------------
# 6) COOKIE MANAGER
# -----------------------------------------------------------------------------
# Provide a password to avoid the 'NoneType' encode error
cookies = EncryptedCookieManager(
    prefix="aiquiz_",
    password="MY_SECURE_PASSWORD"  # <--- replace with your own secret string
)
if not cookies.ready():
    st.stop()

# -----------------------------------------------------------------------------
# 7) MAIN APP
# -----------------------------------------------------------------------------
def add_footer():
    footer = """
    <style>
    /* Make footer fixed at bottom */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f9f9f9;
        text-align: center;
        padding: 0.5rem 0;
        font-size: 0.9rem;
        color: #999999;
    }
    </style>
    <div class="footer">
        <p>All rights reserved @ SASTRA Deemed to be University (Venkatesh T & Team)</p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)
def main():
    st.title("EIE116 Introduction to Artificial Intelligence (One Attempt per Student)")
    st.subheader("First AI based Exam for AI Course in the history of SASTRA.")
    st.write("This is your main content...")

    # Load or init
    if "df_data" not in st.session_state:
        st.session_state["df_data"] = load_data()

    # If cookie is set, block
    if cookies.get("test_submitted") == "true":
        st.error("You have already submitted the test in this browser. No retakes allowed.")
        show_admin_section()
        return

    # Student picks registration number
    reg_no_list = sorted(STUDENTS.keys())
    selected_reg_no = st.selectbox("Select your Registration Number:", reg_no_list)
    st.write(f"**Name:** {STUDENTS[selected_reg_no]}")

    # Student password field
    entered_student_pass = st.text_input("Enter Student Password:", type="password")

    # Start / Resume
    if st.button("Start / Resume Test"):
        # 1) Check student password first
        if entered_student_pass != STUDENT_PASSWORD:
            st.error("Incorrect student password. Please try again.")
            return

        # 2) If password is correct, proceed
        sid = selected_reg_no
        record = get_user_record(st.session_state["df_data"], sid)

        if record and record["status"] == "submitted":
            st.error(f"Student ID '{sid}' has already submitted. No second attempt allowed.")
        else:
            st.session_state["current_student"] = sid
            if not record:
                st.session_state["df_data"] = update_user_record(
                    st.session_state["df_data"],
                    sid,
                    status="in_progress",
                    score=0,
                    answers_list=[""]*len(QUESTIONS)
                )
                save_data(st.session_state["df_data"])
            st.session_state["test_active"] = True

    if "test_active" not in st.session_state:
        st.session_state["test_active"] = False

    if st.session_state["test_active"]:
        current_stu = st.session_state["current_student"]
        record = get_user_record(st.session_state["df_data"], current_stu)
        current_answers = json.loads(record["answers_json"])

        st.info(f"Student ID: **{current_stu}** - Test in progress.")
        st.write(f"Name: **{STUDENTS[current_stu]}**")

        # Display questions
        for i, qdata in enumerate(QUESTIONS):
            st.subheader(f"Question {i+1}")
            st.write(qdata["question"])
            stored_choice = current_answers[i]
            if stored_choice not in qdata["options"]:
                default_ix = 0
            else:
                default_ix = qdata["options"].index(stored_choice)

            user_pick = st.radio(
                label="",
                options=qdata["options"],
                index=default_ix,
                key=f"q_{i}"
            )
            current_answers[i] = user_pick
            st.write("---")

        # Save partial
        st.session_state["df_data"] = update_user_record(
            st.session_state["df_data"],
            current_stu,
            status="in_progress",
            score=0,
            answers_list=current_answers
        )
        save_data(st.session_state["df_data"])

        if st.button("Submit Final Answers"):
            final_score = compute_score(current_answers)
            st.session_state["df_data"] = update_user_record(
                st.session_state["df_data"],
                current_stu,
                status="submitted",
                score=final_score,
                answers_list=current_answers
            )


            # 2) Capture the current local time
            time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 3) Write the timestamp into the same row, e.g. the "submitted_at" column
            df = st.session_state["df_data"]
            idx = df.index[df["student_id"] == current_stu][0]
            df.at[idx, "submitted_at"] = time_now
            # 4) Save the DataFrame
            save_data(df)
            # 5) Set the cookie so the same browser can't retake
            cookies["test_submitted"] = "true"
            cookies.save()

            
##            save_data(st.session_state["df_data"])
##
##            cookies["test_submitted"] = "true"
##            cookies.save()

            st.success("Your answers have been submitted. This browser is now locked for retakes.")
            st.write(f"**Your Score:** {final_score}/{len(QUESTIONS)}")

            # Detailed feedback
            st.write("### Detailed Feedback")
            for i, ans in enumerate(current_answers):
                correct_letter = QUESTIONS[i]["answer"]
                if ans.startswith(correct_letter):
                    st.write(f"Q{i+1}: Correct (You chose {ans[0]})")
                else:
                    st.write(f"Q{i+1}: Incorrect (You chose {ans[0]}), correct = {correct_letter}")

            st.session_state["test_active"] = False

    # Footer
    # Admin side
    show_admin_section()

##def show_admin_section():
##    st.write("---")
##    st.write("## Course Coordinator Access")
##    admin_pass = st.text_input("Enter Admin Password to enable result download:", type="password")
##    if admin_pass == ADMIN_PASSWORD:
##        st.success("Admin password verified. You may download all results.")
##        csv_data = download_all_results(st.session_state["df_data"])
##        st.download_button(
##            label="Download All Results as CSV",
##            data=csv_data,
##            file_name="all_students_results.csv",
##            mime="text/csv"
##        )
##    else:
##        st.info("Enter the correct admin password to enable the download button.")
def show_admin_section():
    # Create two columns: left for text and password input, right for the second logo
    col_left, col_right = st.columns([3, 1])
    
    with col_left:
        st.write("## Course Coordinator Access")
        admin_pass = st.text_input("Enter Admin Password to enable result download:", type="password")
        
        if admin_pass == ADMIN_PASSWORD:
            st.success("Admin password verified. You may download all results.")
            csv_data = download_all_results(st.session_state["df_data"])
            st.download_button(
                label="Download All Results as CSV",
                data=csv_data,
                file_name="all_students_results.csv",
                mime="text/csv"
            )
        else:
            st.info("Enter the correct admin password to enable the download button.")
    
    with col_right:
        # Display the second logo in the right column.
        # Adjust the file name / path as needed.
        st.write("")  
        st.write("")
        st.write("")
        st.write("")
        st.image("sastra_logo1.jpg", use_container_width=True)

if __name__ == "__main__":
    add_footer()
    main()
