import streamlit as st # type: ignore

st.set_page_config(page_title="Churn Prediction App", layout="wide")

# Create tabs for navigation
tab1, tab2, tab3 = st.tabs([":orange[Home]", ":white[Predict]", ":green[Graphs]"])

with tab1:
    import home
    home.app()

with tab2:
    import predict
    predict.app()

with tab3:
    import graph
    graph.app()

