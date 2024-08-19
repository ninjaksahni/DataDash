import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re

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

    # Sidebar for user role adjustment and data cleaning
    with st.sidebar:
        # Column Role Adjustment
        with st.expander("Column Role Adjustment", expanded=False):
            st.write("Adjust the detected column roles if they don't fit your data:")
            for col in df.columns:
                current_role = "date" if col in roles["date"] else "categorical" if col in roles["categorical"] else "numerical"
                new_role = st.selectbox(
                    f"{col} (detected as {current_role})", 
                    ["date", "categorical", "numerical"], 
                    index=["date", "categorical", "numerical"].index(current_role),
                    key=f"{col}_role"
                )
                
                # Update roles based on user selection
                for role in roles:
                    if col in roles[role]:
                        roles[role].remove(col)
                roles[new_role].append(col)
        
            st.write("**Updated Column Roles:**")
            st.write(roles)

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

    # Custom Metrics
    st.write("### Custom Metrics")
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

    # Correlation Analysis
    st.write("### Correlation Analysis")
    if len(roles["numerical"]) >= 2:
        col1 = st.selectbox("Select First Numerical Column", options=roles["numerical"])
        col2 = st.selectbox("Select Second Numerical Column", options=roles["numerical"], index=1)
        if st.button("Calculate Correlation"):
            correlation = df[[col1, col2]].corr().iloc[0, 1]
            st.metric(label="Correlation Coefficient", value=f"{correlation:.2f}")
            
            # Scatter plot showing the correlation
            fig = px.scatter(df, x=col1, y=col2, trendline='ols', title=f"Correlation between {col1} and {col2}")
            st.plotly_chart(fig)
    else:
        st.write("At least two numerical columns are required for correlation analysis.")

    # Adding More Charts
    st.write("### Add More Charts")
    chart_types = ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot"]
    if 'charts' not in st.session_state:
        st.session_state.charts = []
        st.session_state.chart_columns = set()

    with st.form("add_chart_form", clear_on_submit=True):
        st.write("#### Add a New Chart")
        chart_type = st.selectbox("Select Chart Type", chart_types)
        x_axis = st.selectbox("Select X-axis", options=roles["numerical"] + roles["categorical"] + roles["date"], key="x_axis")
        y_axis = st.selectbox("Select Y-axis", options=roles["numerical"], key="y_axis") if chart_type in ["Bar Chart", "Line Chart", "Box Plot"] else None
        values = st.selectbox("Select Values", options=roles["numerical"], key="values") if chart_type == "Pie Chart" else None
        names = st.selectbox("Select Names", options=roles["categorical"], key="names") if chart_type == "Pie Chart" else None
        bins = st.slider("Select Number of Bins", min_value=10, max_value=100, value=20, key="bins") if chart_type == "Histogram" else None

        if st.form_submit_button("Add Chart"):
            chart_key = len(st.session_state.charts) + 1
            st.session_state.charts.append({
                "type": chart_type,
                "x": x_axis,
                "y": y_axis,
                "values": values,
                "names": names,
                "bins": bins
            })
            st.session_state.chart_columns.add(x_axis)

    # Display charts
    if st.session_state.charts:
        st.write("### Generated Charts")
        for idx, chart in enumerate(st.session_state.charts):
            st.write(f"#### Chart {idx + 1}")
            if chart["type"] == "Bar Chart":
                if chart["y"] and chart["x"]:
                    fig = px.bar(df, x=chart["x"], y=chart["y"], title=f"Bar Chart of {chart['y']} vs {chart['x']}")
                    st.plotly_chart(fig)
            elif chart["type"] == "Line Chart":
                if chart["y"] and chart["x"]:
                    fig = px.line(df, x=chart["x"], y=chart["y"], title=f"Line Chart of {chart['y']} vs {chart['x']}")
                    st.plotly_chart(fig)
            elif chart["type"] == "Scatter Plot":
                if chart["y"] and chart["x"]:
                    fig = px.scatter(df, x=chart["x"], y=chart["y"], title=f"Scatter Plot of {chart['y']} vs {chart['x']}")
                    st.plotly_chart(fig)
            elif chart["type"] == "Pie Chart":
                if chart["values"] and chart["names"]:
                    fig = px.pie(df, names=chart["names"], values=chart["values"], title=f"Pie Chart of {chart['values']} by {chart['names']}")
                    st.plotly_chart(fig)
            elif chart["type"] == "Histogram":
                if chart["x"]:
                    fig = px.histogram(df, x=chart["x"], nbins=chart["bins"], title=f"Histogram of {chart['x']}")
                    st.plotly_chart(fig)
            elif chart["type"] == "Box Plot":
                if chart["x"] and chart["y"]:
                    fig = px.box(df, x=chart["x"], y=chart["y"], title=f"Box Plot of {chart['y']} by {chart['x']}")
                    st.plotly_chart(fig)
