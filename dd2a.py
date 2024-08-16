import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Function to detect column types
def detect_column_roles(df):
    roles = {"date": [], "categorical": [], "numerical": []}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            roles["date"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
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
st.title("dataDASH KSV2📊")
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
        color = st.selectbox("Select Color", options=roles["categorical"] + roles["date"], key="color") if chart_type == "Scatter Plot" else None
        
        submit_button = st.form_submit_button("Add Chart")
        if submit_button:
            chart_details = {
                "type": chart_type,
                "x_axis": x_axis,
                "y_axis": y_axis,
                "values": values,
                "names": names,
                "bins": bins,
                "color": color
            }
            st.session_state.charts.append(chart_details)
            st.session_state.chart_columns.update({x_axis, y_axis, values, names})

    st.write("### Existing Charts")
    for idx, chart in enumerate(st.session_state.charts):
        if chart["type"] == "Bar Chart":
            fig = px.bar(df, x=chart["x_axis"], y=chart["y_axis"], title=f"Bar Chart: {chart['y_axis']} vs {chart['x_axis']}")
        elif chart["type"] == "Line Chart":
            fig = px.line(df, x=chart["x_axis"], y=chart["y_axis"], title=f"Line Chart: {chart['y_axis']} vs {chart['x_axis']}")
        elif chart["type"] == "Scatter Plot":
            fig = px.scatter(df, x=chart["x_axis"], y=chart["y_axis"], color=chart["color"], title=f"Scatter Plot: {chart['y_axis']} vs {chart['x_axis']} (Color: {chart['color']})")
        elif chart["type"] == "Pie Chart":
            fig = px.pie(df, values=chart["values"], names=chart["names"], title=f"Pie Chart: {chart['values']} by {chart['names']}")
        elif chart["type"] == "Histogram":
            fig = px.histogram(df, x=chart["x_axis"], nbins=chart["bins"], title=f"Histogram: {chart['x_axis']}")
        elif chart["type"] == "Box Plot":
            fig = px.box(df, x=chart["x_axis"], y=chart["y_axis"], title=f"Box Plot: {chart['y_axis']} by {chart['x_axis']}")
        
        # Display the plot with unique key
        st.plotly_chart(fig, key=f"chart_{idx}")

        # Display insights for each chart
        st.write(f"**Insights for Chart {idx + 1}:**")
        if chart["type"] in ["Bar Chart", "Line Chart", "Box Plot"]:
            st.write(f"- **Mean**: {df[chart['y_axis']].mean():.2f}")
            st.write(f"- **Median**: {df[chart['y_axis']].median():.2f}")
            st.write(f"- **Standard Deviation**: {df[chart['y_axis']].std():.2f}")
            st.write(f"- **Min**: {df[chart['y_axis']].min():.2f}")
            st.write(f"- **Max**: {df[chart['y_axis']].max():.2f}")
        elif chart["type"] == "Pie Chart":
            st.write(f"- **Top Category**: {df[chart['names']].mode()[0]}")
            st.write(f"- **Category Counts**: {df[chart['names']].value_counts().to_dict()}")
        elif chart["type"] == "Histogram":
            st.write(f"- **Mean**: {df[chart['x_axis']].mean():.2f}")
            st.write(f"- **Median**: {df[chart['x_axis']].median():.2f}")
            st.write(f"- **Standard Deviation**: {df[chart['x_axis']].std():.2f}")
            st.write(f"- **Min**: {df[chart['x_axis']].min():.2f}")
            st.write(f"- **Max**: {df[chart['x_axis']].max():.2f}")
