import pandas as pd
import numpy as np
import datetime
import streamlit as st
import altair as alt
from streamlit_autorefresh import st_autorefresh
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from supabase import create_client, Client

# Load environment variables or put directly (not recommended for production)
SUPABASE_URL = "https://xkzgtehagcvzghuupfjm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhremd0ZWhhZ2N2emdodXVwZmptIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc2MDQ5MzEsImV4cCI6MjA3MzE4MDkzMX0.uuoMoqn5VIajJ66aGf2l1_NGAwbzBlr7TW3-KqKbmCw"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Example: fetch tasks
data = supabase.table("tasks").select("*").execute()
print(data)
columns = ["Name", "Urgency", "Importance", "Impact", "DueDate",
           "DaysTillDue", "EstEffort", "ROI", "GoalAlignment", 
           "Consequence", "Status", "Type"]

# ---------------- Data Handling ----------------
@st.cache_data(ttl = 5)

def get_df():
    response = supabase.table("tasks").select("*").execute()
    df = pd.DataFrame(response.data)
    print(df)
    if not df.empty and "DueDate" in df.columns:
        df["DueDate"] = pd.to_datetime(df["DueDate"])
    return df

def save_df(new_task_df):
    for _, row in new_task_df.iterrows():
        supabase.table("tasks").insert(row.to_dict()).execute()


def remove_task(task_name):
    supabase.table("tasks").delete().eq("Name", task_name).execute()

def importance_computer(goal_alignment, impact, consequence_of_neglect, effort, w1=0.4, w2=0.3, w3=0.2, w4=0.1):
    return w1*goal_alignment + w2*impact + w3*consequence_of_neglect + w4*effort

def update_urgency(df):
    if df.empty:
        return df
    dates = pd.to_datetime(df["DueDate"])
    days_till_due = (dates - pd.to_datetime(datetime.date.today())).dt.days
    df["urgency"] = 1/(1+(days_till_due/(df["EstEffort"]*15)))
    return df

def weight_computer(df, w1=0.57, w2=0.3, w3=0.13):
    df_wc = pd.DataFrame({
        "Task Name": df["Name"],
        "Task Weight": (w1*df["Importance"] + w2*df["Urgency"] + w3*(df["Importance"]*df["Urgency"]))
    })
    return df_wc

# ---------------- Streamlit Interface ----------------
st.set_page_config(page_title="Eisenhower Matrix", page_icon="📊", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .quadrant-labels {
        font-size: 14px;
        font-weight: bold;
        color: #333;
        background: rgba(255,255,255,0.8);
        padding: 4px 8px;
        border-radius: 4px;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>📊 Eisenhower Matrix Dashboard</h1></div>', unsafe_allow_html=True)

df = get_df()
print(df)
df = update_urgency(df)

# Initialize session state
if "add_open" not in st.session_state:
    st.session_state["add_open"] = False
if "remove_open" not in st.session_state:
    st.session_state["remove_open"] = False

# Callback functions to toggle expanders
def toggle_add_expander():
    st.session_state["add_open"] = not st.session_state["add_open"]
    if st.session_state["add_open"]:
        st.session_state["remove_open"] = False

def toggle_remove_expander():
    st.session_state["remove_open"] = not st.session_state["remove_open"]
    if st.session_state["remove_open"]:
        st.session_state["add_open"] = False

# Task Management Section
st.markdown("---")
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    if not df.empty:
        total_tasks = len(df)
        st.metric("Total Tasks", total_tasks)
    else:
        st.metric("Total Tasks", 0)

with col2:
    if not df.empty:
        not_started = len(df[df["Status"] == "not started"])
        st.metric("Not Started", not_started)
    else:
        st.metric("Not Started", 0)

with col3:
    if not df.empty:
        in_progress = len(df[df["Status"] == "in progress"])
        st.metric("In Progress", in_progress)
    else:
        st.metric("In Progress", 0)

with col4:
    if not df.empty:
        completed = len(df[df["Status"] == "done"])
        st.metric("Completed", completed)
    else:
        st.metric("Completed", 0)

st.markdown("---")

# Control buttons
col1, col2 = st.columns(2)
with col1:
    st.button("➕ Add New Task", on_click=toggle_add_expander, type="primary")
with col2:
    st.button("🗑 Remove Task", on_click=toggle_remove_expander, type="secondary")

# ---- Add Task Form ----
if st.session_state["add_open"]:
    st.markdown("### ➕ Add New Task")
    with st.form("add_task_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("📝 Task Name", placeholder="Enter task name...")
            impact = st.slider("💥 Impact (0-1)", 0.0, 1.0, 0.5, help="How much impact will this task have?")
            days_till_due = st.number_input("📅 Days Till Due", min_value=0, value=1)
            task_time = st.number_input("⏱️ Estimated Effort (hours)", min_value=0.0, value=1.0)
            
        with col2:
            est_effort = 1 - 1/(1 + task_time)
            goal_alignment = st.slider("🎯 Goal Alignment (0-1)", 0.0, 1.0, 0.5, help="How well does this align with your goals?")
            consequence = st.slider("⚠️ consequence of Neglect (0-1)", 0.0, 1.0, 0.5, help="What happens if you don't do this?")
            status = st.selectbox("📊 status", ["not started", "in progress", "done"])
            task_type = st.text_input("🏷️ Task type", placeholder="e.g., Work, Personal, etc.")

        col1, col2, col3 = st.columns([1, 1, 2])
        with col2:
            submitted = st.form_submit_button("✅ Add Task", type="primary")
        
        if submitted and name:  # Only add if name is provided
            # Save task
            due_date = datetime.date.today() + datetime.timedelta(days=days_till_due)
            urgency = 1/(1+(days_till_due/(est_effort*15)))
            importance = importance_computer(goal_alignment, impact, consequence, est_effort)
            roi = impact/(impact + np.log2(1+est_effort))
            new_task = pd.DataFrame([{
                "Name": name,
                "Urgency": urgency,
                "Importance": importance,
                "Impact": impact,
                "DueDate": due_date.isoformat(),
                "DaysTillDue": days_till_due,
                "EstEffort": est_effort,
                "ROI": roi,
                "GoalAlignment": goal_alignment,
                "Consequence": consequence,
                "Status": status,
                "Type": task_type,
            }])

            save_df(new_task)

            st.success(f"🎉 Task '{name}' added successfully!")
            st.session_state["add_open"] = False
            st.rerun()

# ---- Remove Task Form ----
if st.session_state["remove_open"]:
    st.markdown("### 🗑 Remove Task")
    with st.form("remove_task_form"):
        if not df.empty:
            remove_task_name = st.selectbox("Select Task to Remove", df["Name"])
            col1, col2, col3 = st.columns([1, 1, 2])
            with col2:
                submitted = st.form_submit_button("🗑 Remove Task", type="primary")
            
            if submitted:
                remove_task(remove_task_name)  # <-- call SQLite delete function
                st.success(f"✅ Task '{remove_task_name}' removed successfully!")
                st.session_state["remove_open"] = False
                st.rerun()
        else:
            st.info("📝 No tasks to remove")
            st.form_submit_button(
                "Close",
                on_click=lambda: setattr(st.session_state, "remove_open", False)
            )


st.markdown("---")

if not df.empty:
    # ---- Task Table ----
    st.markdown("### 📋 Current Tasks")
    
    # Create styled dataframe
    display_df = df[["Name", "Status", "Importance", "Urgency", "Impact","ROI", "DueDate", "Type"]].copy()
    display_df["Importance"] = display_df["Importance"].round(3)
    display_df["Urgency"] = display_df["Urgency"].round(3)
    display_df["Impact"] = display_df["Impact"].round(3)
    display_df["ROI"] = display_df["ROI"].round(3)
    
    st.dataframe(display_df, use_container_width=True)

    # ---- Eisenhower Matrix Plot ----
    st.markdown("### 📐 Eisenhower Matrix")
    
    matrix_plot = df.copy()
    
    # Define colors for different statuses
    status_colors = {
        "not started": "#95a5a6",  # Gray
        "in progress": "#3498db",  # Blue
        "done": "#27ae60"          # Green
    }
    
    matrix_plot["Color"] = matrix_plot["Status"].map(status_colors)
    
    # Create the Eisenhower matrix chart
    base = alt.Chart(matrix_plot).add_selection(
        alt.selection_single()
    )
    
    # Background rectangles for quadrants
    quad_data = pd.DataFrame({
        'x': [0, 0.5, 0, 0.5],
        'y': [0, 0, 0.5, 0.5],
        'x2': [0.5, 1, 0.5, 1],
        'y2': [0.5, 0.5, 1, 1],
        'quadrant': ['Not Urgent/Not Important', 'Urgent/Not Important', 
                    'Not Urgent/Important', 'Urgent/Important'],
        'color': ['#f8f9fa', '#fff3cd', '#d1ecf1', '#f5c6cb']
    })
    
    quadrants = alt.Chart(quad_data).mark_rect(
        opacity=0.3,
        stroke='#6c757d',
        strokeWidth=1
    ).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[0, 1])),
        x2=alt.X2('x2:Q'),
        y2=alt.Y2('y2:Q'),
        color=alt.Color('color:N', scale=None)
    )
    
    # Task points
    points = base.mark_circle(
        size=300,
        stroke='white',
        strokeWidth=2
    ).encode(
        x=alt.X("Urgency:Q", 
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(title="Urgency →", titleFontSize=14, labelFontSize=12)),
        y=alt.Y("Importance:Q", 
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(title="↑ Importance", titleFontSize=14, labelFontSize=12)),
        color=alt.Color("Color:N", scale=None, legend=None),
        tooltip=["Name:N", "Status:N", "Importance:Q", "Urgency:Q", "Impact:Q", "Type:N"]
    )
    
    # Quadrant labels
    label_data = pd.DataFrame({
        'x': [0.75, 0.25, 0.75, 0.25],
        'y': [0.75, 0.75, 0.25, 0.25],
        'text': ['DO FIRST\n(Important & Urgent)', 'DECIDE\n(Important, Not Urgent)', 
                'DELEGATE\n(Not Important, Urgent)', 'DELETE\n(Not Important, Not Urgent)'],
        'color': ['#dc3545', '#28a745', '#fd7e14', '#6c757d']
    })
    
    labels = alt.Chart(label_data).mark_text(
        align='center',
        baseline='middle',
        fontSize=11,
        fontWeight='bold'
    ).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[0, 1])),
        text='text:N',
        color=alt.Color('color:N', scale=None)
    )
    
    # Combine all layers
    eisenhower_chart = (quadrants + points + labels).resolve_scale(
        color='independent'
    ).properties(
        width=600,
        height=700,
        title=alt.TitleParams("Tasks positioned by importance vs urgency", fontSize=16, anchor='start')
    )
    
    st.altair_chart(eisenhower_chart, use_container_width=True)
    
    # Legend
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("⚫ **Not Started**")  
    with col2:
        st.markdown("🔵 **In Progress**")
    with col3:
        st.markdown("🟢 **Completed**")

    # ---- Weighted Task Priority ----
    st.markdown("### 🏆 Task Priority (Weighted Score)")
    weights_df = weight_computer(df).sort_values("Task Weight", ascending=False)
    
    # Ensure Task Weight is clamped to 0-1 range
    weights_df["Task Weight"] = weights_df["Task Weight"].clip(0, 1)
    
    bar_chart = alt.Chart(weights_df).mark_bar(
        color='#667eea',
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    ).encode(
        x=alt.X("Task Name:N", 
                sort=alt.EncodingSortField(field="Task Weight", order="descending"),
                axis=alt.Axis(title="Tasks", labelAngle=-45, titleFontSize=14)),
        y=alt.Y("Task Weight:Q", 
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(title="Priority Score", titleFontSize=14)),
        tooltip=["Task Name:N", alt.Tooltip("Task Weight:Q", format=".3f")]
    ).properties(
        height=400,
        title=alt.TitleParams("Task Priority Ranking (Higher = More Important)", fontSize=16, anchor='start')
    )

    st.altair_chart(bar_chart, use_container_width=True)
    
else:
    # Empty state
    st.markdown("### 🌟 Welcome to Your Eisenhower Matrix!")
    st.info("👆 Click 'Add New Task' above to get started with organizing your tasks by importance and urgency.")
    
    # Show example matrix
    st.markdown("### 📖 How the Eisenhower Matrix Works:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🔴 DO FIRST (Important & Urgent)**
        - Crises and emergencies
        - Deadline-driven projects
        - Last-minute preparations
        """)
        
        st.markdown("""
        **🟠 DELEGATE (Not Important, Urgent)**
        - Interruptions
        - Some meetings
        - Some phone calls
        """)
    
    with col2:
        st.markdown("""
        **🟢 DECIDE (Important, Not Urgent)**
        - Planning and strategy
        - Learning and development
        - Exercise and health
        """)
        
        st.markdown("""
        **⚫ DELETE (Not Important, Not Urgent)**
        - Time wasters
        - Excessive social media
        - Unnecessary activities
        """)
