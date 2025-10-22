import pandas as pd
import numpy as np
import datetime
import streamlit as st
import altair as alt
from streamlit_autorefresh import st_autorefresh
import psycopg2
from dotenv import load_dotenv
import os
from supabase import create_client, Client
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import re
from streamlit_cookies import CookieManager
import json

# Load environment variables
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except KeyError as e:
    st.error(f"❌ Missing configuration: {str(e)}. Please check your secrets.")
    st.stop()
except Exception as e:
    st.error(f"❌ Could not connect to Supabase: {str(e)}")
    st.stop()


def validate_password(password):
    """Validate password complexity"""
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not re.search(r"\d", password):
        errors.append("Password must contain at least one number")
    
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        errors.append("Password must contain at least one special character")
    
    return errors

# Initialize cookie manager
cookies = CookieManager()

# Authentication Section with Cookie Persistence
if "user" not in st.session_state:
    st.session_state["user"] = None

# Check for auth cookie on page load
if st.session_state["user"] is None:
    auth_cookie = cookies.get("supabase_auth")
    if auth_cookie:
        try:
            auth_data = json.loads(auth_cookie)
            
            # Validate with Supabase using stored tokens
            response = supabase.auth.set_session(auth_data["access_token"], auth_data["refresh_token"])
            if response.user:
                st.session_state["user"] = response.user
        except Exception:
            # Clear invalid cookie
            cookies.delete("supabase_auth")

# Professional Authentication UI
if st.session_state["user"] is None:
    # Create a centered authentication container with improved CSS
    st.markdown("""
    <style>
    .auth-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 30vh;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: -1rem -1rem 2rem -1rem;
        padding: 2rem;
        border-radius: 0 0 20px 20px;
    }
    .auth-box {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        width: 100%;
        max-width: 450px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .auth-title {
        color: #667eea;
        margin-bottom: 1.5rem;
        font-size: 3.5rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .auth-subtitle {
        color: #6c757d;
        margin-bottom: 2rem;
        font-size: 1.1rem;
        font-weight: 400;
    }
    </style>
    
    <div class="auth-container">
        <div class="auth-box">
            <h1 class="auth-title">🎯MyFlow</h1>
            <p class="auth-subtitle">Your Productivity Command Center</p>
    """, unsafe_allow_html=True)

    # Authentication tabs with enhanced styling
    auth_tab1, auth_tab2 = st.tabs(["🔐 Sign In", "✨ Create Account"])
    
    with auth_tab1:
        with st.form("login_form", clear_on_submit=False):
            st.markdown("### Welcome Back!")
            st.markdown("Sign in to access your productivity dashboard")
            
            email = st.text_input("📧 Email Address", placeholder="Enter your email")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                remember_me = st.checkbox("Remember me")
            with col2:
                st.markdown('<p style="text-align: right; margin: 0;"><a href="#" style="color: #667eea; text-decoration: none; font-size: 0.9rem;">Forgot password?</a></p>', unsafe_allow_html=True)
            
            login_btn = st.form_submit_button("🚀 Sign In", type="primary", use_container_width=True)
            
            if login_btn:
                if email and password:
                    try:
                        with st.spinner("Signing you in..."):
                            res = supabase.auth.sign_in_with_password({
                                "email": email,
                                "password": password
                            })
                            st.session_state["user"] = res.user
                            
                            # Store auth data in cookie for persistence (no expiration params)
                            if remember_me and res.session:
                                auth_data = {
                                    "access_token": res.session.access_token,
                                    "refresh_token": res.session.refresh_token
                                }
                                cookies.set("supabase_auth", json.dumps(auth_data))
                            
                            st.success("✅ Welcome back!")
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Sign in failed: {str(e)}")
                else:
                    st.error("❌ Please fill in all fields")
    
    with auth_tab2:
        with st.form("signup_form", clear_on_submit=False):
            st.markdown("### Join MyFlow Today!")
            st.markdown("Create your account and start optimizing your productivity")
            
            email = st.text_input("📧 Email Address", placeholder="Enter your email")
            password = st.text_input("🔒 Password", type="password", placeholder="Create a strong password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")
            
            # Password requirements info
            with st.expander("📋 Password Requirements"):
                st.markdown("""
                - At least 8 characters long
                - One uppercase letter (A-Z)
                - One lowercase letter (a-z)
                - One number (0-9)
                - One special character (!@#$%^&*...)
                """)
            
            terms_accepted = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            signup_btn = st.form_submit_button("✨ Create Account", type="primary", use_container_width=True)
            
            if signup_btn:
                if email and password and confirm_password:
                    if password != confirm_password:
                        st.error("❌ Passwords don't match")
                    else:
                        # Validate password complexity
                        password_errors = validate_password(password)
                        if password_errors:
                            for error in password_errors:
                                st.error(f"❌ {error}")
                        elif not terms_accepted:
                            st.error("❌ Please accept the terms and conditions")
                        else:
                            try:
                                with st.spinner("Creating your account..."):
                                    res = supabase.auth.sign_up({
                                        "email": email,
                                        "password": password
                                    })
                                    if res.user:
                                        st.success("🎉 Account created successfully! Please check your email for verification, then head to the sign in page.")
                                    else:
                                        st.success("🎉 Account created successfully! Please sign in.")
                                    time.sleep(2)
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Account creation failed: {str(e)}")
                else:
                    st.error("❌ Please fill in all fields")
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hide the main content when not authenticated
    st.stop()

else:
    # User is logged in - show user info in sidebar
    with st.sidebar:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        ">
            <h3 style="margin: 0; font-size: 1.2rem;">👋 Welcome back!</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">{}</p>
        </div>
        """.format(st.session_state['user'].email), unsafe_allow_html=True)
        
        st.info("💡 **Tip:** Use 'R' key to refresh instead of Ctrl+R to stay logged in")

        if st.button("🚪 Sign Out", use_container_width=True):
            supabase.auth.sign_out()
            cookies.delete("supabase_auth")  # Clear the auth cookie
            st.session_state["user"] = None
            st.rerun()

def update_days_till_due_for_all_tasks():
    """Update DaysTillDue AND Urgency for all user's tasks in the database"""
    if st.session_state["user"] is None:
        return False
    
    try:
        # Get all user tasks with required fields
        response = supabase.table("tasks").select("Name, DueDate, EstEffort").eq("user_id", st.session_state["user"].id).execute()
        
        today = datetime.date.today()
        update_count = 0
        
        for task in response.data:
            if task.get("DueDate") and task.get("EstEffort"):
                due_date = pd.to_datetime(task["DueDate"]).date()
                days_till_due = max(0, (due_date - today).days)
                
                # Calculate urgency
                urgency = round(1 / (1 + (days_till_due / (task["EstEffort"] * 5))), 3)
                
                # Update both fields in database
                supabase.table("tasks").update({
                    "DaysTillDue": days_till_due,
                    "Urgency": urgency
                }).eq("Name", task["Name"]).eq("user_id", st.session_state["user"].id).execute()
                
                update_count += 1
        
        clear_data_cache()
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to update days till due: {str(e)}")
        return False


#Insights Engine
class Productivity:
    @staticmethod
    def analyze_task_patterns(df):
        if df.empty:
            return {}
        
        insights = {}
        
        # Time allocation analysis
        urgent_important = len(df[(df['Urgency'] > 0.7) & (df['Importance'] > 0.7)])
        insights['crisis_mode'] = urgent_important / len(df) > 0.3
        
        # Productivity score
        completed_high_impact = len(df[(df['Status'] == 'done') & (df['Impact'] > 0.7)])
        total_high_impact = len(df[df['Impact'] > 0.7])
        insights['productivity_score'] = completed_high_impact / max(total_high_impact, 1)
        
        # Procrastination detection
        overdue_important = len(df[(df['DaysTillDue'] < 0) & (df['Importance'] > 0.6)])
        insights['procrastination_risk'] = overdue_important > 0
        
        # Time allocation recommendation
        quadrant_distribution = {
            'urgent_important': len(df[(df['Urgency'] > 0.5) & (df['Importance'] > 0.5)]),
            'important_not_urgent': len(df[(df['Urgency'] <= 0.5) & (df['Importance'] > 0.5)]),
            'urgent_not_important': len(df[(df['Urgency'] > 0.5) & (df['Importance'] <= 0.5)]),
            'neither': len(df[(df['Urgency'] <= 0.5) & (df['Importance'] <= 0.5)])
        }
        insights['quadrant_distribution'] = quadrant_distribution
        
        return insights
    
    @staticmethod
    def generate_recommendations(insights, df):
        recommendations = []
        
        if insights.get('crisis_mode', False):
            recommendations.append("🚨 You're in crisis mode! 30%+ of tasks are urgent & important. Focus on prevention strategies.")
        
        if insights.get('productivity_score', 0) < 0.5:
            recommendations.append("📈 Your high-impact completion rate is below 50%. Consider breaking down complex tasks.")
        
        if insights.get('procrastination_risk', False):
            recommendations.append("⏰ You have overdue important tasks. Schedule focused work blocks immediately.")
        
        # Optimal time allocation suggestion
        quad_dist = insights.get('quadrant_distribution', {})
        ideal_important_not_urgent = quad_dist.get('important_not_urgent', 0)
        if ideal_important_not_urgent < len(df) * 0.6:
            recommendations.append("🎯 Increase 'Important but Not Urgent' tasks to 60%+ for long-term success.")
        
        return recommendations

@dataclass
class TaskMetrics:
    total_tasks: int
    completion_rate: float
    avg_importance: float
    avg_urgency: float
    burnout_risk: float

# Enhanced Data Handling
@st.cache_data(ttl=60)  # Changed from 5 seconds to 60 seconds
def get_df():
    if st.session_state["user"] is None:
        return pd.DataFrame()

    try:
        response = supabase.table("tasks").select("*").eq("user_id", st.session_state["user"].id).execute()
        df = pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"❌ Failed to fetch tasks: {str(e)}")
        return pd.DataFrame()

    if not df.empty:
        if "DueDate" in df.columns:
            df["DueDate"] = pd.to_datetime(df["DueDate"], errors="coerce")
        
        # CRITICAL FIX: Always recalculate urgency from fresh data
        if "EstEffort" in df.columns and "DueDate" in df.columns:
            today = pd.to_datetime(datetime.date.today())
            days_till_due = (df["DueDate"] - today).dt.days
            days_till_due = days_till_due.clip(lower=0)  # Prevent negative values
            df["Urgency"] = 1 / (1 + (days_till_due / (df["EstEffort"] * 5)))
            df["Urgency"] = df["Urgency"].round(3)
    
    return df


def clear_data_cache():
    """Clear the cached data to force refresh"""
    get_df.clear()


def save_df(new_task_df):
    if st.session_state["user"] is None:
        return False

    try:
        existing_tasks = supabase.table("tasks").select("Name").eq("user_id", st.session_state["user"].id).execute()
        existing_names = {task["Name"] for task in existing_tasks.data}
    except Exception as e:
        st.error(f"❌ Could not check existing tasks: {str(e)}")
        return False

    for _, row in new_task_df.iterrows():
        task_name = row["Name"]
        if task_name in existing_names:
            st.error(f"❌ Task name '{task_name}' already exists.")
            return False

        try:
            row_dict = row.to_dict()
            row_dict["user_id"] = st.session_state["user"].id
            
            # CRITICAL FIX: Ensure urgency is calculated correctly before saving
            if "DueDate" in row_dict and "EstEffort" in row_dict:
                due_date = pd.to_datetime(row_dict["DueDate"])
                today = pd.to_datetime(datetime.date.today())
                days_till_due = max(0, (due_date - today).days)
                row_dict["Urgency"] = round(1 / (1 + (days_till_due / (row_dict["EstEffort"] * 5))), 3)
                row_dict["DaysTillDue"] = days_till_due
            
            supabase.table("tasks").insert(row_dict).execute()
        except Exception as e:
            st.error(f"❌ Failed to save task '{task_name}': {str(e)}")
            return False

    # Clear cache to show new data immediately
    clear_data_cache()
    return True


def update_task_status(task_name, new_status):
    if st.session_state["user"] is None:
        return False
        
    try:
        # Get current task data for urgency recalculation
        current_task = supabase.table("tasks").select("*").eq("Name", task_name).eq("user_id", st.session_state["user"].id).single().execute()
        
        if current_task.data:
            # Recalculate urgency based on current date
            task_data = current_task.data
            if task_data.get("DueDate") and task_data.get("EstEffort"):
                due_date = pd.to_datetime(task_data["DueDate"])
                today = pd.to_datetime(datetime.date.today())
                days_till_due = max(0, (due_date - today).days)
                new_urgency = round(1 / (1 + (days_till_due / (task_data["EstEffort"] * 5))), 3)
                
                # Update both status and urgency
                supabase.table("tasks").update({
                    "Status": new_status,
                    "Urgency": new_urgency,
                    "DaysTillDue": days_till_due
                }).eq("Name", task_name).eq("user_id", st.session_state["user"].id).execute()
            else:
                # Just update status if no date/effort data
                supabase.table("tasks").update({"Status": new_status}).eq("Name", task_name).eq("user_id", st.session_state["user"].id).execute()
        
        clear_data_cache()
        return True
        
    except Exception as e:
        st.error(f"❌ Could not update task '{task_name}': {str(e)}")
        return False

def remove_task(task_name):
    if st.session_state["user"] is None:
        return
    try:
        supabase.table("tasks").delete().eq("Name", task_name).eq("user_id", st.session_state["user"].id).execute()
        # Clear cache to show updated data immediately
        clear_data_cache()
    except Exception as e:
        st.error(f"❌ Could not delete task '{task_name}': {str(e)}")


def importance_computer(goal_alignment, impact, consequence_of_neglect, effort, w1=0.4, w2=0.3, w3=0.2, w4=0.1):
    return w1*goal_alignment + w2*impact + w3*consequence_of_neglect + w4*effort


def update_urgency(df):
    if df.empty:
        return df
    df = df.copy()
    dates = pd.to_datetime(df["DueDate"])
    days_till_due = (dates - pd.to_datetime(datetime.date.today())).dt.days
    df["urgency"] = 1/(1+(days_till_due/(df["EstEffort"]*5)))
    df["Urgency"] = df["urgency"]
    return df


def weight_computer(df, w1=0.57, w2=0.3, w3=0.13):
    if df.empty:
        return pd.DataFrame(columns=["Task Name", "Task Weight"])
    return pd.DataFrame({
        "Task Name": df["Name"],
        "Task Weight": (w1*df["Importance"] + w2*df["Urgency"] + w3*(df["Importance"]*df["Urgency"]))
    })


def calculate_metrics(df) -> TaskMetrics:
    if df.empty:
        return TaskMetrics(0, 0, 0, 0, 0)
    
    total = len(df)
    completed = len(df[df["Status"] == "done"])
    completion_rate = round(completed / total, 2) if total > 0 else 0
    avg_importance = round(df["Importance"].mean(), 2)
    avg_urgency = round(df["Urgency"].mean(), 2)
    
    urgent_important = len(df[(df["Urgency"] > 0.7) & (df["Importance"] > 0.7)])
    burnout_risk = round(min(urgent_important / max(total, 1), 1.0), 2)
    
    return TaskMetrics(total, completion_rate, avg_importance, avg_urgency, burnout_risk)


# Streamlit Interface
st.set_page_config(
    page_title="Productivity Command Center", 
    page_icon="🎯", 
    layout="wide"
)

# Clean CSS
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
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    .insight-card {
        background: linear-gradient(135deg, #ffeaa7, #fab1a0);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #e17055;
        color: #2d3436;
        font-weight: 500;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #a8e6cf, #88d8a3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #00b894;
        color: #2d3436;
        font-weight: 500;
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
    .quadrant-label {
        font-size: 12px;
        font-weight: bold;
        color: #333;
        background: rgba(255,255,255,0.8);
        padding: 4px 8px;
        border-radius: 4px;
    }
    .tab-content {
        padding: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🎯 Productivity Command Center</h1></div>', unsafe_allow_html=True)

# Load and process data
df = get_df()
# Auto-update days till due once per session
if "days_updated_today" not in st.session_state:
    st.session_state["days_updated_today"] = False

if not st.session_state["days_updated_today"]:
    if update_days_till_due_for_all_tasks():
        st.session_state["days_updated_today"] = True
    
@st.cache_data(ttl=60)
def get_processed_data(df):
    """Process dataframe with current calculations - no caching to avoid sync issues"""
    if df.empty:
        return df.copy(), pd.DataFrame()
    
    df_processed = df.copy()
    
    # Ensure urgency is current (this should already be done in get_df, but double-check)
    if "DueDate" in df_processed.columns and "EstEffort" in df_processed.columns:
        today = pd.to_datetime(datetime.date.today())
        dates = pd.to_datetime(df_processed["DueDate"])
        days_till_due = (dates - today).dt.days.clip(lower=0)
        df_processed["Urgency"] = (1 / (1 + (days_till_due / (df_processed["EstEffort"] * 5)))).round(3)
    
    # Calculate weights
    if not df_processed.empty:
        weights_df = pd.DataFrame({
            "Task Name": df_processed["Name"],
            "Task Weight": (0.57 * df_processed["Importance"] + 
                          0.3 * df_processed["Urgency"] + 
                          0.13 * (df_processed["Importance"] * df_processed["Urgency"]))
        }).sort_values("Task Weight", ascending=False)
    else:
        weights_df = pd.DataFrame(columns=["Task Name", "Task Weight"])
    
    return df_processed, weights_df


df = get_df()

# Auto-update days till due once per session
if "days_updated_today" not in st.session_state:
    st.session_state["days_updated_today"] = False

if not st.session_state["days_updated_today"]:
    if update_days_till_due_for_all_tasks():
        st.session_state["days_updated_today"] = True

# Process data without problematic caching
df, weights_df = get_processed_data(df)

# Initialize session state
if "add_open" not in st.session_state:
    st.session_state["add_open"] = False

# Sidebar filters
with st.sidebar:
    st.markdown("## Filters")
    
    # Time-based filtering
    time_filter = st.selectbox("Time Horizon", 
        ["All Tasks", "Today", "This Week", "This Month", "This Quarter"])
    
    # Priority filtering
    priority_filter = st.slider("Min Importance", 0.0, 1.0, 0.0, 0.1)
    
    # Category filtering
    if not df.empty:
        categories = ["All"] + list(df["Type"].dropna().unique())
        category_filter = st.selectbox("Category", categories)
    else:
        category_filter = "All"

# Apply filters
filtered_df = df.copy()
if not filtered_df.empty:
    if time_filter != "All Tasks":
        if time_filter == "Today":
            filtered_df = filtered_df[filtered_df["DaysTillDue"] == 0]
        elif time_filter == "This Week":
            filtered_df = filtered_df[filtered_df["DaysTillDue"] <= 7]
        elif time_filter == "This Month":
            filtered_df = filtered_df[filtered_df["DaysTillDue"] <= 30]
        elif time_filter == "This Quarter":
            filtered_df = filtered_df[filtered_df["DaysTillDue"] <= 90]
    
    filtered_df = filtered_df[filtered_df["Importance"] >= priority_filter]
    
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df["Type"] == category_filter]

# Tabs
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📐 Matrix", "⚡ Actions", "🧠 Analytics"])


# Set active tab based on session state
if st.session_state.active_tab == 0:  # Dashboard tab
    st.session_state.active_tab = 0

if st.session_state.active_tab == 1:  # Matrix tab
    st.session_state.active_tab = 1

if st.session_state.active_tab == 2:  # Actions tab
    st.session_state.active_tab = 2

if st.session_state.active_tab == 3:  # Analytics tab
    st.session_state.active_tab = 3


with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    # Key metrics
    metrics = calculate_metrics(filtered_df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.total_tasks}</div>
            <div class="metric-label">Total Tasks</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.completion_rate:.1%}</div>
            <div class="metric-label">Completion Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.avg_importance:.2f}</div>
            <div class="metric-label">Avg Importance</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.avg_urgency:.2f}</div>
            <div class="metric-label">Avg Urgency</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        risk_color = "#e74c3c" if metrics.burnout_risk > 0.7 else "#f39c12" if metrics.burnout_risk > 0.4 else "#27ae60"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {risk_color}">{metrics.burnout_risk:.1%}</div>
            <div class="metric-label">Burnout Risk</div>
        </div>
        """, unsafe_allow_html=True)

    #Insights
    if not filtered_df.empty:
        insights = Productivity.analyze_task_patterns(filtered_df)
        recommendations = Productivity.generate_recommendations(insights, filtered_df)
        
        st.markdown("### 🤖 Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Performance Analysis")
            productivity_score = insights.get('productivity_score', 0)
            if productivity_score >= 0.8:
                st.markdown(f"<div class='recommendation-card'>🌟 Exceptional: {productivity_score:.1%} high-impact completion</div>", unsafe_allow_html=True)
            elif productivity_score >= 0.6:
                st.markdown(f"<div class='insight-card'>📈 Good: {productivity_score:.1%} high-impact completion</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='insight-card'>⚠️ Below target: {productivity_score:.1%} completion rate</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Recommendations")
            for rec in recommendations:
                st.markdown(f"<div class='recommendation-card'>{rec}</div>", unsafe_allow_html=True)

    # Task overview table
    if not filtered_df.empty:
        st.markdown("### 📋 Task Overview")
        
        display_df = filtered_df[["Name", "Status", "Type", "Importance", "Urgency", "Impact", "ROI", "DueDate"]].copy()
        
        for col in ["Importance", "Urgency", "Impact", "ROI", "GoalAlignment", "Consequence", "EstEffort"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        st.dataframe(display_df, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    if not filtered_df.empty:
        # Original Eisenhower Matrix (Altair style)
        st.markdown("### 📐 Eisenhower Matrix")
        
        # Status colors
        status_colors = {
            "not started": "#95a5a6",
            "in progress": "#3498db", 
            "done": "#27ae60"
        }

        matrix_plot = filtered_df.copy()
        matrix_plot["Color"] = matrix_plot["Status"].map(status_colors)
        matrix_plot["id"] = matrix_plot.index.astype(str)

        
        matrix_plot["Color"] = matrix_plot["Status"].map(status_colors)
        
        # Create chart
        base = alt.Chart(matrix_plot).add_selection(alt.selection_single())
        
        # Background quadrants
        quad_data = pd.DataFrame({
            'x': [0, 0.5, 0, 0.5],
            'y': [0, 0, 0.5, 0.5],
            'x2': [0.5, 1, 0.5, 1],
            'y2': [0.5, 0.5, 1, 1],
            'quadrant': ['Delete', 'Delegate', 'Decide', 'Do'],
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
                    axis=alt.Axis(title="Urgency →")),
            y=alt.Y("Importance:Q", 
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(title="↑ Importance")),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Name:N", "Status:N", "Importance:Q", "Urgency:Q", "Impact:Q", "Type:N"],
            key="id:N"
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
        
        # Combine layers
        eisenhower_chart = (quadrants + points + labels).resolve_scale(
            color='independent'
        ).properties(
            width=600,
            height=600,
            title="Tasks positioned by importance vs urgency"
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

        # Priority ranking
        st.markdown("### 🏆 Priority Rankings")
        if not filtered_df.empty and not weights_df.empty:
            filtered_df_not_done_tasks = filtered_df[filtered_df["Status"] != "done"].copy()
            filtered_weights = weights_df[weights_df["Task Name"].isin(filtered_df_not_done_tasks["Name"])].copy()
            filtered_weights["Task Weight"] = filtered_weights["Task Weight"].clip(0, 1)
            filtered_weights = filtered_weights.sort_values("Task Weight", ascending=False)
        else:
            filtered_weights = pd.DataFrame(columns=["Task Name", "Task Weight"])
        
        bar_chart = alt.Chart(filtered_weights).mark_bar(
            color='#667eea',
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x=alt.X("Task Name:N", 
                    sort=alt.EncodingSortField(field="Task Weight", order="descending"),
                    axis=alt.Axis(title="Tasks", labelAngle=-45)),
            y=alt.Y("Task Weight:Q", 
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(title="Priority Score")),
            tooltip=["Task Name:N", alt.Tooltip("Task Weight:Q", format=".3f")]
        ).properties(
            height=400,
            title="Task Priority Ranking"
        )

        st.altair_chart(bar_chart, use_container_width=True)
    
    else:
        st.info("Add some tasks to see your Eisenhower Matrix!")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    # Add Task Form
    st.markdown("### ➕ Add New Task")
    with st.form("add_task_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            name = st.text_input("📝 Task Name", placeholder="Enter task name...")
            impact = st.slider("💥 Impact", 0.0, 1.0, 0.5, 0.1)
            days_till_due = st.number_input("📅 Days Till Due", min_value=0, value=7)
            
        with col2:
            task_time = st.number_input("⏱️ Estimated Hours", min_value=0.1, value=2.0, step=0.5)
            goal_alignment = st.slider("🎯 Goal Alignment", 0.0, 1.0, 0.5, 0.1)
            consequence = st.slider("⚠️ Consequence of Delay", 0.0, 1.0, 0.5, 0.1)
            
        with col3:
            status = st.selectbox("📊 Status", ["not started", "in progress", "done"])
            task_type = st.selectbox("🏷️ Category", 
                ["Strategic", "Operational", "Administrative", "Personal", "Innovation", "Crisis"])

        submitted = st.form_submit_button("✅ Add Task", type="primary")
        
        if submitted and name:
            est_effort = 1 - 1/(1 + task_time)
            due_date = datetime.date.today() + datetime.timedelta(days=days_till_due)
            if days_till_due < 0:
                days_till_due = 0
            urgency = round(1 / (1 + (days_till_due / (est_effort * 5))), 2)
            importance = round(importance_computer(goal_alignment, impact, consequence, est_effort), 2)
            roi = round(impact / (impact + np.log2(1 + est_effort)), 2)
            
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
                "Type": task_type
            }])

            if save_df(new_task):
                st.success(f"🎉 Task '{name}' added successfully!")
                st.session_state.active_tab = 2  # Keep on Actions tab (0-indexed)
                time.sleep(1)
                st.rerun()

    st.markdown("---")

    # Quick Actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🗑 Remove Task")
        
        # Initialize session state
        if "remove_task_success" not in st.session_state:
            st.session_state.remove_task_success = False
        if "form_submitted_successfully" not in st.session_state:
            st.session_state.form_submitted_successfully = False
        
        if not filtered_df.empty:
            # Use the success flag to control form clearing
            clear_form = st.session_state.form_submitted_successfully
            
            with st.form("remove_task_form", clear_on_submit=clear_form):
                remove_task_name = st.selectbox("Select Task to Remove", filtered_df["Name"])
                confirm_delete = st.checkbox("⚠️ I confirm I want to delete this task")
                submitted = st.form_submit_button("🗑 Remove Task", type="secondary")
                
                if submitted:
                    if confirm_delete:
                        remove_task(remove_task_name)
                        st.session_state.remove_task_success = True
                        st.session_state.removed_task_name = remove_task_name
                        st.session_state.form_submitted_successfully = True
                        st.rerun()
                    else:
                        # Reset the success flag to prevent form clearing
                        st.session_state.form_submitted_successfully = False
                        st.error("❌ Please confirm deletion by checking the checkbox")
            
            # Show success message and reset form state
            if st.session_state.remove_task_success:
                st.success(f"✅ Task '{st.session_state.removed_task_name}' removed!")
                st.session_state.remove_task_success = False
                st.session_state.form_submitted_successfully = False  # Reset for next time
        else:
            st.info("No tasks to remove")

            
    with col2:
        st.markdown("### ⚡ Bulk Update")
        
        # Initialize session state for bulk update
        if "bulk_tasks_selected" not in st.session_state:
            st.session_state.bulk_tasks_selected = []
        if "bulk_status_selected" not in st.session_state:
            st.session_state.bulk_status_selected = "not started"
        if "bulk_update_success" not in st.session_state:
            st.session_state.bulk_update_success = False
        
        if not filtered_df.empty:
            # Create form to prevent auto-rerun
            with st.form("bulk_update_form"):
                tasks_to_update = st.multiselect(
                    "Select Tasks", 
                    filtered_df["Name"],
                    default=st.session_state.bulk_tasks_selected if st.session_state.bulk_tasks_selected else []
                )
                new_status = st.selectbox(
                    "New Status", 
                    ["not started", "in progress", "done"],
                    index=["not started", "in progress", "done"].index(st.session_state.bulk_status_selected)
                )
                
                submitted = st.form_submit_button("🔄 Update Selected")
                
                if submitted and tasks_to_update:
                    for task in tasks_to_update:
                        update_task_status(task, new_status)
                    
                    # Update session state - keep the selected tasks
                    st.session_state.bulk_tasks_selected = []  # Keep selection after update
                    st.session_state.bulk_status_selected = new_status  # Keep the selected status
                    st.session_state.bulk_update_success = True
                    
                    st.rerun()
            
            # Show success message outside the form
            if st.session_state.bulk_update_success:
                st.success(f"✅ Tasks updated successfully!")
                st.session_state.bulk_update_success = False  # Reset the flag
                
        else:
            st.info("No tasks available")
    
    with col3:
        st.markdown("### 📊 Export")
        if not filtered_df.empty:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "tasks_export.csv",
                "text/csv",
                type="secondary"
            )
        else:
            st.info("No data to export")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    if not filtered_df.empty:
        # Personal optimization
        st.markdown("### 🧠 Personal Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚡ Energy Management")
            high_impact_tasks = len(filtered_df[filtered_df['Impact'] > 0.7])
            critical_tasks = len(filtered_df[filtered_df['PriorityLevel'] == 'Critical']) if 'PriorityLevel' in filtered_df.columns else 0
            
            st.metric("High-Impact Tasks", high_impact_tasks)
            st.metric("Critical Tasks", critical_tasks)
            
            if critical_tasks > 3:
                st.warning("⚠️ Too many critical tasks - consider re-prioritizing")
            else:
                st.success("✅ Manageable critical task load")
        
        with col2:
            st.markdown("#### 🎯 Focus Optimization")
            deep_work_tasks = filtered_df[
                (filtered_df['EstEffort'] > 0.6) & (filtered_df['Importance'] > 0.6)
            ]
            shallow_tasks = filtered_df[
                (filtered_df['EstEffort'] <= 0.3) | (filtered_df['Importance'] <= 0.4)
            ]
            
            st.metric("Deep Work Sessions", len(deep_work_tasks))
            st.metric("Quick Wins", len(shallow_tasks))
            
            if len(deep_work_tasks) > 0:
                st.success(f"📚 Schedule {len(deep_work_tasks)} deep work blocks")
            
            if len(shallow_tasks) > 5:
                st.info(f"⚡ Batch {len(shallow_tasks)} quick tasks together")

        # Analytics charts
        st.markdown("### 💡 Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Time allocation by category
            category_effort = filtered_df.groupby('Type')['EstEffort'].sum().sort_values(ascending=False)
            if len(category_effort) > 0:
                time_fig = px.pie(
                    values=category_effort.values,
                    names=category_effort.index,
                    title="Time Investment by Category"
                )
                st.plotly_chart(time_fig, use_container_width=True)
        
        with col2:
            # Completion quality
            completed_df = filtered_df[filtered_df['Status'] == 'done']
            if not completed_df.empty:
                avg_completed_importance = completed_df['Importance'].mean()
                
                st.metric("Completion Quality Score", f"{avg_completed_importance:.2f}")
                
                if avg_completed_importance > 0.7:
                    st.success("🌟 You're completing high-value tasks!")
                elif avg_completed_importance > 0.5:
                    st.info("📈 Good focus on important work")
                else:
                    st.warning("🎯 Focus more on high-importance tasks")

        # Advanced analytics
        if len(filtered_df) >= 5:
            st.markdown("### 🔮 Predictive Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Completion forecast
                completed_tasks = len(filtered_df[filtered_df['Status'] == 'done'])
                remaining_tasks = len(filtered_df[filtered_df['Status'] != 'done'])
                
                if completed_tasks > 0:
                    completion_velocity = completed_tasks / len(filtered_df)
                    days_to_complete = remaining_tasks / max(completion_velocity * len(filtered_df) / 7, 0.1)
                    
                    forecast_date = datetime.date.today() + datetime.timedelta(days=int(days_to_complete))
                    st.success(f"📈 Projected completion: {forecast_date.strftime('%B %d, %Y')}")
                else:
                    st.warning("⏳ Start completing tasks to generate forecast")
            
            with col2:
                # Optimization opportunities
                low_importance = filtered_df[filtered_df['Importance'] < 0.3]
                high_effort_low_impact = filtered_df[(filtered_df['EstEffort'] > 0.7) & (filtered_df['Impact'] < 0.4)]
                
                if len(low_importance) > 0:
                    st.info(f"🗑️ {len(low_importance)} tasks could be eliminated")
                
                if len(high_effort_low_impact) > 0:
                    st.warning(f"⚡ {len(high_effort_low_impact)} high-effort, low-impact tasks need review")

        # Comprehensive analytics
        with st.expander("📊 Advanced Analytics", expanded=False):
            if len(filtered_df) >= 3:
                fig_analytics = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Task Distribution by Status', 'Importance vs Urgency',
                                  'ROI Distribution', 'Effort vs Impact'),
                    specs=[[{'type': 'pie'}, {'type': 'scatter'}],
                           [{'type': 'histogram'}, {'type': 'scatter'}]]
                )
                
                # Status distribution
                status_counts = filtered_df['Status'].value_counts()
                fig_analytics.add_trace(
                    go.Pie(labels=status_counts.index, values=status_counts.values, name="Status"),
                    row=1, col=1
                )
                
                # Importance vs Urgency
                fig_analytics.add_trace(
                    go.Scatter(x=filtered_df['Urgency'], y=filtered_df['Importance'],
                              mode='markers', text=filtered_df['Name'],
                              marker=dict(size=10, opacity=0.7), name="Tasks"),
                    row=1, col=2
                )
                
                # ROI distribution
                fig_analytics.add_trace(
                    go.Histogram(x=filtered_df['ROI'], nbinsx=10, name="ROI"),
                    row=2, col=1
                )
                
                # Effort vs Impact
                fig_analytics.add_trace(
                    go.Scatter(x=filtered_df['EstEffort'], y=filtered_df['Impact'],
                              mode='markers', text=filtered_df['Name'],
                              marker=dict(size=10, opacity=0.7), name="Effort vs Impact"),
                    row=2, col=2
                )
                
                fig_analytics.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig_analytics, use_container_width=True)

    else:
        st.info("Add some tasks to see analytics and insights!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("**System Status:** 🟢 Operational | " + datetime.datetime.now().strftime("%H:%M:%S"))
