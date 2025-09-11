import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import streamlit as st

path = "eisenhower_matrix\matrix_data.csv"



def get_df():
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError: 
            pd.DataFrame({
            "Name": [],
            "Urgency": [],
            "Importance": [],
            "Impact": [],
            "Due Date": [],
            "Days Till Due": [],
            "Est. effort": [],
            "ROI": [],
            "Goal Alignment" : [],
            "Consequence": [],
            "Status": [],
            "Type": [],
            }).to_csv(path, index = False)
            df = pd.read_csv(path)
    return df


def importance_computer(goal_alignment, impact, consequence_of_neglect, effort,  w1 = 0.4,w2 = 0.3,w3 = 0.2,w4 = 0.1):
    return w1*goal_alignment + w2*impact + w3*consequence_of_neglect + w4*effort 


def update_urgency(df):
    dates = pd.to_datetime(df["Due Date"]).copy()
    Days_Till_Due = (dates - pd.to_datetime(datetime.date.today())).dt.days
    print(Days_Till_Due)
    df["Urgency"] = 1/(1+(Days_Till_Due/df["Est. effort"]*15))
    df.to_csv(path, index=False)
    
    
def weight_computer(df, w1 = 0.57, w2 = 0.3, w3 = 0.13):
    weights ={
            "Task Name": [],
            "Task Weight":[]
        }
    weights["Task Name"] = list(df["Name"])
    for i in range(len(df)):
        weight_algorithm = (w1*df["Importance"][i]) + (w2*df["Urgency"][i]) + (w3*(df["Importance"][i]*df["Urgency"][i]))
        weights["Task Weight"].append(weight_algorithm)
    return pd.DataFrame(weights)


def add(df, Name,Urgency,Importance,Impact,due_date,days_till_due, Est_effort, ROI, goal_alignment,consequence,Status,Type):
    add_df = pd.DataFrame({
        "Name": [Name],
        "Urgency": [Urgency],
        "Importance": [Importance],
        "Impact": [Impact],
        "ROI": [ROI],
        "Due Date": [due_date],
        "Days Till Due": [days_till_due],
        "Est. effort": [Est_effort],
        "Goal Alignment" : [goal_alignment],
        "Consequence": [consequence],
        "Status": [Status],
        "Type": [Type],
        })
    df = pd.concat((df, add_df), axis= 0).reset_index(drop=True)
    df.to_csv(path, index = False)
    return df

def remove(df, name):
    if name not in df["Name"].values:
        print("Task not found.")
        return
    df.drop(df[df["Name"] == name].index, axis=0, inplace=True)
    df.to_csv(path, index = False)
    return df

def edit(df, name):
    if name not in [i.lower() for i in df["Name"]]:
        print("Task not found.")
        return
    idx = df[df["Name"].str.lower() == name]
    print(idx)
    while True:
        feature = input("Feature: ")
        if feature == "break":
            return
        if feature in df.columns:
            break
    old_value = df.at[idx.index[0], feature]
    while True:
        try:
            if isinstance(old_value, (float, np.floating)):
                value = float(input("New value: "))
                break
            elif isinstance(old_value, (bool, np.bool_)):
                value = bool(input("New value: "))
                break
            else:
                value = input("New value: ").lower()
                if value == "break":
                    return
                break
        except ValueError:
            print("Invalid. Enter a valid input.")
    df[feature].replace(old_value, value, inplace = True)
    df.to_csv(path, index = False)


def matricizer(df):
    colors = np.where(df["Status"] == "In progress", 'blue','grey')
    fig, (plt1, plt2) = plt.subplots(1,2,figsize = (11,6), gridspec_kw={"width_ratios": [2,1]})
    plt1.axis((1,0,0,1))
    plt1.set_xticks(np.arange(0,1.1,0.1))
    plt1.set_yticks(np.arange(0,1.1,0.1))
    plt1.set_xlabel(tuple(df.keys())[1])
    plt1.set_ylabel(tuple(df.keys())[2])
    plt1.axhline(0.5, c = 'k')
    plt1.axvline(0.5, c = 'k')
    x = df["Urgency"]
    y = df["Importance"]
    plt1.scatter(x, y, c= colors)
    plt1.grid(True)
    for i, label in enumerate(df["Name"]):
        plt1.annotate(label, (x[i], y[i]), textcoords= "offset points",xytext= (0,5),ha = "center", c = "red")
    
    df1 = weight_computer(df)
    df1.sort_values("Task Weight", ascending=False, inplace=True)
    x1, y1 = df1["Task Name"], df1["Task Weight"]
    plt2.bar(x1,y1)
    plt2.set_yticks(np.arange(0,1.1,0.1))
    plt2.grid(axis= "y")
    plt.tight_layout()
    plt.show()

"""def streamlit_view(df):
    st.title("Eisenhower Matrix")

    # Scatter matrix
    st.subheader("Urgency vs Importance")
    st.scatter_chart(df, x="Urgency", y="Importance", color="Status", size="Impact")

    # Weighted priority bar chart
    df1 = weight_computer(df).sort_values("Task Weight", ascending=False)
    st.subheader("Task Priorities")
    st.bar_chart(df1.set_index("Task Name"))
"""

def interface():
    while True:
        df = get_df()
        update_urgency(df)
        try:
            while True:
                action = int(input("1: View\n2: Add\n3: Remove\n4: Edit\n>"))
                if action < 5:
                    break
                print("Integers must be within the option range.")
            if action == 0:
                break    
            while True:
                if action == 1:
                    matricizer(df)
                    break
                elif action == 2:
                    df = get_df()
                    Name = input("Name: ").lower()
                    Impact = float(input("Impact (0-1): "))
                    Days_Till_Due = float(input("days till due: "))
                    while True:
                        task_time = float(input("estimated effort (hours): "))
                        if task_time >= 0:
                            break
                        print("Zero or less are not a valid input.")
                    Est_effort = 1-1/(1+task_time)
                    goal_alignment = float(input("Goal alignment (0-1): "))
                    consequence = float(input("Consequence of neglect (0-1): "))
                    Status = "not started"
                    Type = input("Task type: ").lower()
                    due_date = datetime.date.today() + datetime.timedelta(days=Days_Till_Due)
                    ROI = Impact/(Impact + np.log2(1+Est_effort))
                    Urgency = 1/(1+(Days_Till_Due/Est_effort*15))
                    Importance = importance_computer(goal_alignment, Impact, consequence, Est_effort) 
                    add(df, Name,Urgency,Importance, Impact,due_date,Days_Till_Due,Est_effort,ROI,goal_alignment,consequence,Status,Type)
                elif action == 3:
                    print(df["Name"].head(len(df)))
                    name = input("Task name: ").lower()
                    if name == "break":
                        break
                    remove(df, name)
                elif action == 4:
                    print(df["Name"].head(len(df)))
                    name = input("Task name: ").lower()
                    if name == "break":
                        break
                    edit(df, name)
                else:
                    break
                print("\n")
        except ValueError:
            print("Invalid input.")
    print("Program Terminated")

interface()