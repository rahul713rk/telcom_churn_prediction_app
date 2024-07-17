import pandas as pd
import streamlit as st
import plotly.express as px

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('app/dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    data['SeniorCitizen'] = data['SeniorCitizen'].map({1:'Yes' , 0 : 'No'})
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
    data.drop(columns={'customerID'} , inplace=True)
    return data

def app():
    st.title("Interactive Graphs Section")

    data = load_data()

    # Fixed X-axis column
    y_col = "Churn"

    # Select Y-axis column
    x_col = st.selectbox("Select X-axis column", data.columns)

    # Checkbox for hue option
    use_hue = st.checkbox("Use hue (categorical color separation)", value=False)

    # Select hue column if checkbox is checked
    hue_col = None
    if use_hue:
        hue_col = st.selectbox("Select hue column", [None] + list(data.columns))

    # Select graph type
    graph_type = st.selectbox("Select graph type", ["Bar", "Scatter", "Line", "Box", "Violin", "Pie"])

    # Plot based on selections
    if graph_type == "Bar":
        if y_col:
            fig = px.bar(data, x=x_col, y=y_col, color=hue_col, title=f'Bar Chart of {y_col} vs {x_col}')
            st.plotly_chart(fig)

    elif graph_type == "Scatter":
        if y_col:
            fig = px.scatter(data, x=x_col, y=y_col, color=hue_col, title=f'Scatter Plot of {y_col} vs {x_col}')
            st.plotly_chart(fig)

    elif graph_type == "Line":
        if y_col:
            fig = px.line(data, x=x_col, y=y_col, color=hue_col, title=f'Line Chart of {y_col} vs {x_col}')
            st.plotly_chart(fig)

    elif graph_type == "Box":
        fig = px.box(data, x=x_col, y=y_col, color=hue_col, title=f'Box Plot of {y_col} vs {x_col}')
        st.plotly_chart(fig)

    elif graph_type == "Violin":
        fig = px.violin(data, x=x_col, y=y_col, color=hue_col, title=f'Violin Plot of {y_col} vs {x_col}')
        st.plotly_chart(fig)

    elif graph_type == "Pie":
        fig = px.pie(data, names=x_col, title=f'Pie Chart of {x_col}', color=hue_col)
        st.plotly_chart(fig)
