import streamlit as st # type: ignore

def app():
    st.title("Telcom Churn :red[Prediction App]")

    st.header("Welcome to the Churn Prediction App!")
    st.write(
        "This application helps you predict customer churn using machine learning. "
        "Explore the functionalities of the tabs below:"
    )

    st.subheader("Functionality Overview:")
    st.markdown("""
    - **Predict**: Use our churn prediction model to see if a customer is likely to churn based on their details.
    - **Graphs**: Visualize data trends and customer behavior through interactive graphs.
    """)

    st.subheader("Dataset Information")
    st.write(
        "The dataset used for this application is the Telco Customer Churn dataset from Kaggle. "
        "[Dataset Link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)"
    )

    st.markdown("""
    **Dataset Description:**
    - The dataset contains information about customers, including demographic details and account information.
    - It includes the target variable, indicating whether a customer has churned or not.
    - Key features include:
      - Customer ID
      - Gender
      - Age
      - Monthly Charges
      - Tenure
      - Contract type
      - Payment method
      - And more...
    """)

    st.subheader("How to Use This App")
    st.write(
        "1. Navigate to the **Predict** tab to input customer details for churn prediction."
        "\n2. Use the **Graphs** tab to analyze customer behavior visually."
    )

    st.write("Thank you for using the Churn Prediction App!")