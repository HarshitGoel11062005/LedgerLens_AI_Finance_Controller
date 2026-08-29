
import streamlit as st
import pandas as pd
import io
import os
from google import genai

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)
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

            st.subheader("🤖 AI Financial Investigation")

        if st.button("Investigate Financial Data"):

          investigation_prompt = f"""
You are LedgerLens, an AI financial investigation assistant.

Analyze ONLY the financial evidence provided below.

Rules:
1. Identify suspicious or abnormal financial activity.
2. Explain the evidence clearly.
3. Calculate or mention financial impact when supported by the evidence.
4. Do not invent missing information.
5. If evidence is insufficient, explicitly say so.
6. Do not authorize or execute financial transactions.
7. Recommendations are for human review only.
8. Be concise and professional.

Return the investigation using exactly:

Finding:
Evidence:
Financial Impact:
Possible Cause:
Recommendation:
Confidence:

FINANCIAL EVIDENCE:
{extracted_text}
"""

        with st.spinner("AI is investigating the financial data..."):

            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=investigation_prompt
                )

                st.success("Investigation completed!")

                st.subheader("📋 Investigation Report")

                st.text_area(
                    "LedgerLens AI Report",
                    response.text,
                    height=500
                )

            except Exception as e:
                st.error(f"AI investigation failed: {e}")

        else:
            st.warning(
                "No readable text was found in this PDF."
            )

    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
