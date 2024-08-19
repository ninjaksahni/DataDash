import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re

# Initialize session state for charts
if "charts" not in st.session_state:
    st.session_state.charts = []

# Function to clean and convert currency or formatted strings to numerical values
def clean_and_convert_currency(df):
    for col in df.columns:
        if df[col].dtype == object:
            # Check if the column contains currency-like values using regex
            if df[col].apply(lambda x: isinstance(x, str) and bool(re.match(r'^\D*\d{1,3}(,\d{3})*(\.\d+)?$', x.strip()))).all():
                # Remove currency symbols and commas, and convert to float
                df[col] = df[col].apply(lambda x: float(re.sub(r'[^\d.]', '', x)))
    return df

# Function to detect column types and handle special cases
def detect_column_roles(df):
    df = clean_and_convert_currency(df)  # Clean currency-like columns
    roles = {"date": [], "categorical": [], "numerical": []}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            roles["date"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            roles["numerical"].append(col)
        elif df[col].apply(lambda x: isinstance(x, str) and "%" in x).all():
            # Convert percentage strings to numerical values
            df[col] = df[col].str.rstrip('%').astype('float') / 100.0
            roles["numerical"].append(col)
        else:
            roles["categorical"].append(col)
    return roles

# Data Cleaning Functions
def handle_missing_values(df, strategy):
    if strategy == "Drop Rows":
        df = df.dropna()
    elif strategy == "Fill with Mean":
        df = df.fillna(df.mean())
    elif strategy == "Fill with Median":
        df = df.fillna(df.median())
    elif strategy == "Fill with Mode":
        df = df.fillna(df.mode().iloc[0])
    return df

def handle_outliers(df, cols, method):
    for col in cols:
        if method == "Z-Score":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < 3]
        elif method == "IQR":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
    return df

# Streamlit App
st.title("DATA DASH 📊 KSV3")
st.write("Upload a CSV or Excel file to generate a dashboard.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # Read the file into a DataFrame
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Display the DataFrame
    st.write("Here is your data:")
    st.write(df)

    # Detect initial column roles
    roles = detect_column_roles(df)

    # Sidebar for auto correlation analysis and custom metrics
    with st.sidebar:
        st.write("## Auto Correlation Analysis")

        # Compute correlation matrix
        if roles["numerical"]:
            corr_matrix = df[roles["numerical"]].corr()
            corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
            filtered_pairs = corr_pairs[(corr_pairs > 0.8) & (corr_pairs < 0.95)]

            with st.expander("Correlations between 0.8 and 0.95", expanded=False):
                if not filtered_pairs.empty:
                    for (col1, col2), value in filtered_pairs.items():
                        st.write(f"{col1} and {col2}: {value:.2f}")
                else:
                    st.write("No correlations between 0.8 and 0.95 found.")
        else:
            st.write("No numerical columns available for correlation analysis.")

        # Custom Metrics
        with st.expander("Custom Metrics", expanded=False):
            if roles["numerical"]:
                st.write("Define custom metrics for numerical data:")
                col_metric = st.selectbox("Select Column for Metric Calculation", options=roles["numerical"])
                metrics = {
                    "Mean": df[col_metric].mean(),
                    "Median": df[col_metric].median(),
                    "Standard Deviation": df[col_metric].std(),
                    "Max": df[col_metric].max(),
                    "Min": df[col_metric].min()
                }
                for metric, value in metrics.items():
                    st.metric(label=metric, value=f"{value:.2f}")
            else:
                st.write("No numerical columns available for custom metrics.")

        # Handling Missing Values
        with st.expander("Handle Missing Values", expanded=False):
            st.write("Missing values can be handled in several ways:")
            st.write("- **Drop Rows**: Removes any rows with missing values.")
            st.write("- **Fill with Mean**: Replaces missing values with the mean of the column.")
            st.write("- **Fill with Median**: Replaces missing values with the median of the column.")
            st.write("- **Fill with Mode**: Replaces missing values with the mode (most common value) of the column.")
            strategy = st.selectbox("Select Strategy", ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"])
            if st.button("Apply Missing Values Strategy"):
                df = handle_missing_values(df, strategy)

        # Handling Outliers
        with st.expander("Handle Outliers", expanded=False):
            st.write("Outliers are extreme values that differ significantly from other observations. Handling them can improve the analysis.")
            st.write("- **Z-Score**: Identifies outliers based on standard deviations from the mean.")
            st.write("- **IQR (Interquartile Range)**: Identifies outliers based on the spread of the middle 50% of the data.")
            cols = st.multiselect("Select Columns to Check for Outliers", options=df.columns)
            method = st.selectbox("Select Outlier Detection Method", ["Z-Score", "IQR"])
            if st.button("Apply Outliers Strategy"):
                df = handle_outliers(df, cols, method)

    # Data Overview
    with st.expander("Data Overview", expanded=False):
        st.write("### Data Overview")
        st.write("Here is a quick overview of your data:")
        st.write("**Data Types:**")
        st.write(df.dtypes)
        st.write("**First Few Rows:**")
        st.write(df.head())

    # Summary Statistics
    with st.expander("Summary Statistics", expanded=False):
        st.write("### Summary Statistics")
        if roles["numerical"]:
            st.write("**Numerical Columns Summary:**")
            st.write(df[roles["numerical"]].describe())
        else:
            st.write("No numerical columns available for summary statistics.")

    # Charting Section
    st.write("### Add More Charts")
    with st.form("add_chart_form", clear_on_submit=True):
        st.write("#### Add a New Chart")
        chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot"])
        x_axis = st.selectbox("Select X-axis", options=roles["numerical"] + roles["categorical"] + roles["date"], key="x_axis")
        y_axis = st.selectbox("Select Y-axis", options=roles["numerical"], key="y_axis") if chart_type in ["Bar Chart", "Line Chart", "Box Plot"] else None
        values = st.selectbox("Select Values", options=roles["numerical"], key="values") if chart_type == "Pie Chart" else None
        names = st.selectbox("Select Names", options=roles["categorical"], key="names") if chart_type == "Pie Chart" else None
        chart_title = st.text_input("Chart Title", value="My Chart")
        
        submitted = st.form_submit_button("Add Chart")
        if submitted:
            if x_axis and (y_axis or values):
                fig = None
                if chart_type == "Bar Chart":
                    fig = px.bar(df, x=x_axis, y=y_axis, title=chart_title)
                elif chart_type == "Line Chart":
                    fig = px.line(df, x=x_axis, y=y_axis, title=chart_title)
                elif chart_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_axis, y=y_axis, title=chart_title)
                elif chart_type == "Pie Chart":
                    fig = px.pie(df, names=names, values=values, title=chart_title)
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_axis, title=chart_title)
                elif chart_type == "Box Plot":
                    fig = px.box(df, x=x_axis, y=y_axis, title=chart_title)
                
                if fig:
                    fig.update_layout(width=800, height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store the chart in session state
                    st.session_state.charts.append(fig)
                else:
                    st.error("Invalid chart configuration.")
            else:
                st.error("Please select valid columns for the chart.")

    # Display previously added charts
    st.write("### Previously Added Charts")
    for idx, chart in enumerate(st.session_state.charts):
        st.write(f"**Chart {idx + 1}:**")
        st.plotly_chart(chart, use_container_width=True)

    # Allow users to remove charts
    st.write("### Remove a Chart")
    if st.session_state.charts:
        remove_idx = st.selectbox("Select Chart to Remove", options=range(len(st.session_state.charts)))
        if st.button("Remove Selected Chart"):
            st.session_state.charts.pop(remove_idx)
            st.experimental_rerun()
