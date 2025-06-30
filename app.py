# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="EduNest | Satisfaction Analytics", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    return pd.read_csv("EduNest_Synthetic_Dataset_Cleaned.csv")

df = load_data()

# Sidebar Filters
with st.sidebar:
    st.image("https://i.imgur.com/xmGdz4I.png", width=220)
    st.title("EduNest Filters")
    st.markdown("Use filters to interactively explore the dashboard.")
    region = st.multiselect("📍 Region", df["Region"].unique(), default=df["Region"].unique())
    gender = st.multiselect("👤 Gender", df["Gender"].unique(), default=df["Gender"].unique())
    device = st.multiselect("💻 Preferred Device", df["Preferred_Device"].unique(), default=df["Preferred_Device"].unique())

filtered_df = df[
    (df["Region"].isin(region)) & 
    (df["Gender"].isin(gender)) & 
    (df["Preferred_Device"].isin(device))
]

st.title("🎓 EduNest | Customer Satisfaction Dashboard")
st.markdown("##### Gain insights into customer behavior, satisfaction, and engagement. Ideal for management decisions.")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🔍 Visual Explorer", "👥 Segment Analysis", "📈 Correlations", "📂 Raw Data"
])

with tab1:
    st.subheader("💼 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Users", df.shape[0])
    col2.metric("✅ Signup %", f"{(df['Signup_Status']=='Yes').mean()*100:.2f}%")
    col3.metric("📚 Enrollment %", f"{(df['Course_Enrolled']=='Yes').mean()*100:.2f}%")
    col4.metric("🌟 Avg. Satisfaction", f"{df['Satisfaction_Score'].mean():.2f}/10")

    st.markdown("### 🧭 Satisfaction Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["Satisfaction_Score"], bins=20, kde=True, color="#2a9d8f", ax=ax)
    ax.set_title("Histogram of Customer Satisfaction", fontsize=12)
    st.pyplot(fig)

    st.markdown("### 🎯 Avg Satisfaction by Course")
    avg_course = filtered_df.groupby("Course_Interested")["Satisfaction_Score"].mean().sort_values()
    fig = px.bar(avg_course, title="Average Satisfaction by Course")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📊 Visual Explorer")

    st.markdown("#### 🌍 Region vs Satisfaction (Boxplot)")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, x="Region", y="Satisfaction_Score", palette="pastel", ax=ax)
    st.pyplot(fig)

    st.markdown("#### 📈 Signup Status Count")
    fig = px.bar(filtered_df["Signup_Status"].value_counts(), title="Signup Status")
    st.plotly_chart(fig)

    st.markdown("#### 📣 Ad Channel Influence")
    ad_mean = filtered_df.groupby("Ad_Channel")["Satisfaction_Score"].mean().sort_values()
    fig = px.bar(ad_mean, title="Avg Satisfaction by Ad Channel", labels={"value": "Satisfaction Score", "Ad_Channel": "Channel"})
    st.plotly_chart(fig)

    st.markdown("#### 🧬 Age vs Satisfaction (Violin Plot)")
    fig = px.violin(filtered_df, y="Satisfaction_Score", x="Gender", box=True, points="all", color="Gender")
    st.plotly_chart(fig)

with tab3:
    st.subheader("👥 Segment-wise Breakdown")

    st.markdown("#### 🔢 Gender vs Plan Type")
    gender_plan = pd.crosstab(filtered_df["Gender"], filtered_df["Plan_Type"])
    st.dataframe(gender_plan)

    st.markdown("#### 🎓 Completion Status vs Satisfaction")
    comp_df = filtered_df[filtered_df["Course_Completion_Status"] != "N/A"]
    fig = px.box(comp_df, x="Course_Completion_Status", y="Satisfaction_Score", color="Course_Completion_Status")
    st.plotly_chart(fig)

    st.markdown("#### 🎯 Interest Score vs Satisfaction")
    fig = px.scatter(filtered_df, x="Interest_Score", y="Satisfaction_Score", color="Course_Enrolled", trendline="ols")
    st.plotly_chart(fig)

    st.markdown("#### 👥 Referral Count Impact")
    fig = px.box(filtered_df, x="Referral_Count", y="Satisfaction_Score", points="all", color_discrete_sequence=["indianred"])
    st.plotly_chart(fig)

with tab4:
    st.subheader("📈 Correlation Analysis")

    st.markdown("#### 🔗 Heatmap of Numeric Correlations")
    corr = filtered_df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.markdown("#### 🕒 Signup Trends Over Time")
    signup_df = filtered_df[filtered_df["Signup_Date"] != "Not Signed Up"].copy()
    signup_df["Signup_Date"] = pd.to_datetime(signup_df["Signup_Date"])
    signup_count = signup_df.groupby("Signup_Date").size()
    st.line_chart(signup_count)

with tab5:
    st.subheader("📂 Full Dataset View")
    st.dataframe(filtered_df, use_container_width=True)
