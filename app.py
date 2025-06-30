# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="EduNest Satisfaction Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("EduNest_Synthetic_Dataset.csv")

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Apply Filters")
regions = st.sidebar.multiselect("Select Region", options=df["Region"].unique(), default=df["Region"].unique())
gender = st.sidebar.multiselect("Select Gender", options=df["Gender"].unique(), default=df["Gender"].unique())
device = st.sidebar.multiselect("Preferred Device", options=df["Preferred_Device"].unique(), default=df["Preferred_Device"].unique())

filtered_df = df[(df["Region"].isin(regions)) & (df["Gender"].isin(gender)) & (df["Preferred_Device"].isin(device))]

# --- Header ---
st.title("🎓 EduNest: Customer Satisfaction Insights Dashboard")
st.markdown("This interactive dashboard provides detailed insights into customer behavior and satisfaction. Use the filters to customize your view and uncover patterns across regions, devices, gender, and more.")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visual Analysis", "🧠 Advanced Insights", "📍Raw Data"])

with tab1:
    st.subheader("Key Metrics Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", df.shape[0])
    col2.metric("Signup Rate", f"{(df['Signup_Status']=='Yes').mean()*100:.2f}%")
    col3.metric("Enrollment Rate", f"{(df['Course_Enrolled']=='Yes').mean()*100:.2f}%")
    col4.metric("Avg Satisfaction Score", f"{df['Satisfaction_Score'].mean():.2f}/10")

    st.markdown("### Satisfaction Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["Satisfaction_Score"], bins=10, kde=True, ax=ax)
    ax.set_title("Distribution of Customer Satisfaction")
    st.pyplot(fig)

with tab2:
    st.subheader("Behavioral Visuals")

    st.markdown("#### Satisfaction Score by Region")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, x="Region", y="Satisfaction_Score", ax=ax)
    st.pyplot(fig)

    st.markdown("#### Gender-wise Course Enrollment")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x="Gender", hue="Course_Enrolled", ax=ax)
    st.pyplot(fig)

    st.markdown("#### Ad Channel Influence")
    fig, ax = plt.subplots()
    sns.barplot(data=filtered_df, x="Ad_Channel", y="Satisfaction_Score", estimator=np.mean, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown("#### Satisfaction Score by Education Level")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, x="Education_Level", y="Satisfaction_Score", ax=ax)
    st.pyplot(fig)

    st.markdown("#### Monthly Fee Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["Monthly_Fee"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("#### Referral Impact on Satisfaction")
    fig, ax = plt.subplots()
    sns.barplot(data=filtered_df, x="Referral_Count", y="Satisfaction_Score", ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader("Correlations & Completion Insights")

    st.markdown("#### Correlation Heatmap (Numeric Features)")
    numeric_cols = filtered_df.select_dtypes(include=np.number)
    corr = numeric_cols.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown("#### Completion vs Satisfaction")
    if "Course_Completion_Status" in filtered_df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_df.dropna(subset=["Course_Completion_Status"]),
                    x="Course_Completion_Status", y="Satisfaction_Score", ax=ax)
        st.pyplot(fig)

    st.markdown("#### Interest Score vs Satisfaction")
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df, x="Interest_Score", y="Satisfaction_Score", hue="Course_Enrolled", ax=ax)
    st.pyplot(fig)

with tab4:
    st.subheader("Raw Data View")
    st.dataframe(filtered_df, use_container_width=True)
