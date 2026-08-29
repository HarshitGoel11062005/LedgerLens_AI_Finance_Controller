
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

from pypdf import PdfReader

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")

    st.subheader("📄 PDF Information")
    st.write(f"File name: {uploaded_file.name}")

    # Extract text from PDF
    try:
        reader = PdfReader(uploaded_file)

        extracted_text = ""

        for page in reader.pages:
            text = page.extract_text()

            if text:
                extracted_text += text + "\n"

        if extracted_text.strip():
            st.success("PDF text extracted successfully!")

            st.subheader("📑 Extracted Financial Data")

            st.text_area(
                "PDF Content",
                extracted_text,
                height=400
            )

        else:
            st.warning(
                "No readable text was found in this PDF."
            )

    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
