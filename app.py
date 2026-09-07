"""
Task 1: Streamlit-Based Exploratory Data Analysis Interface
CLO-2 | Week 4 - GUI & Problem-Solving Agent

Run locally with:  streamlit run app.py
Deploy on Streamlit Community Cloud (https://share.streamlit.io) by pushing
this file (and a requirements.txt with streamlit, pandas, matplotlib, seaborn)
to a GitHub repo and connecting it there.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Exploratory Data Analysis Interface", layout="wide")

st.title("Exploratory Data Analysis Interface")

# ---------------------------------------------------------------------------
# SIDEBAR: Dataset Controls (file upload + attribute selection)
# ---------------------------------------------------------------------------
st.sidebar.header("Dataset Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File for Analysis", type=["csv"], help="Limit 200MB per file - CSV"
)

if uploaded_file is not None:
    # ---- Validate the file is a proper CSV -------------------------------
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.sidebar.error("The uploaded CSV file is empty.")
            st.stop()
    except Exception as e:
        st.sidebar.error(f"Invalid CSV file: {e}")
        st.stop()

    st.sidebar.success(f"Loaded: {uploaded_file.name}")

    # ---- Attribute Selection ----------------------------------------------
    st.sidebar.header("Attribute Selection")
    selected_col = st.sidebar.selectbox("Select Attribute for Visualization", df.columns)

    # =========================================================================
    # MAIN CONTENT AREA - TOP SECTION: Dataset preview + metadata
    # =========================================================================
    st.header("Dataset Preview & Metadata")

    st.subheader("First 5 Rows:")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Shape:")
    st.write(df.shape)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Column Data Types:")
        dtypes_df = pd.DataFrame(df.dtypes.astype(str), columns=["Data Type"])
        st.dataframe(dtypes_df, use_container_width=True)

    with col2:
        st.subheader("Missing Values per Column:")
        missing_df = pd.DataFrame({
            "Missing Count": df.isnull().sum(),
            "Missing %": (df.isnull().sum() / len(df) * 100).round(4),
        })
        st.dataframe(missing_df, use_container_width=True)

    st.subheader("Statistical Summary (Numerical Attributes):")
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        summary = numeric_df.agg(["mean", "median", "min", "max"]).T
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("No numerical columns found in this dataset.")

    # =========================================================================
    # MAIN CONTENT AREA - BOTTOM SECTION: Conditional visualization canvas
    # =========================================================================
    st.header("Visualization")

    is_numeric = pd.api.types.is_numeric_dtype(df[selected_col])

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if is_numeric:
        # Numerical -> Histogram
        sns.histplot(df[selected_col].dropna(), kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Histogram of {selected_col}")
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Frequency")
    else:
        # Categorical -> Bar chart of frequency counts
        counts = df[selected_col].value_counts()
        percentages = (counts / counts.sum() * 100).round(1)
        bars = ax.bar(counts.index.astype(str), counts.values, color="salmon")
        ax.set_title(f"Bar Chart of {selected_col}")
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        for bar, pct in zip(bars, percentages):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{pct}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    st.pyplot(fig)

else:
    st.info("Upload a CSV file from the sidebar to begin exploratory data analysis. "
            "(Test with titanic.csv as required by the assignment.)")
