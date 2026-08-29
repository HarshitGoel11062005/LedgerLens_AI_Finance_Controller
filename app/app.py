
import streamlit as st
import pandas as pd
import io

st.set_page_config(
    page_title="LedgerLens",
    page_icon="💰",
    layout="wide"
)

st.title("LedgerLens")
st.subheader("AI Finance Controller")

st.write(
    "Detect financial anomalies, investigate suspicious transactions, "
    "and generate AI-powered explanations."
)

st.divider()

st.header("📂 Upload Financial Data")

uploaded_file = st.file_uploader(
    "Upload your financial data",
    type=["csv", "pdf"]
)

if uploaded_file is not None:

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):

        try:
            df = pd.read_csv(uploaded_file)

            st.success("CSV uploaded successfully!")

            st.subheader("📊 Uploaded Data")
            st.dataframe(df, use_container_width=True)

            st.info(f"Rows: {len(df)} | Columns: {len(df.columns)}")

        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    elif file_name.endswith(".pdf"):

        st.success("PDF uploaded successfully!")

        st.subheader("📄 PDF Information")

        st.write("File name:", uploaded_file.name)
        st.write(
            "PDF upload is ready. Text extraction will be connected "
            "in the next step."
        )
