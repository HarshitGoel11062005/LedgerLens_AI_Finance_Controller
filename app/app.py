import streamlit as st
import pandas as pd
from pypdf import PdfReader
from google import genai


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="LedgerLens",
    page_icon="💰",
    layout="wide"
)


# ==========================================
# GEMINI CLIENT
# ==========================================

try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )
except Exception as e:
    client = None


# ==========================================
# HEADER
# ==========================================

st.title("LedgerLens")
st.subheader("AI Finance Controller")

st.write(
    "Detect financial anomalies, investigate suspicious transactions, "
    "and generate AI-powered explanations."
)

st.divider()


# ==========================================
# FILE UPLOAD
# ==========================================

st.header("📂 Upload Financial Data")

uploaded_file = st.file_uploader(
    "Upload your financial data",
    type=["csv", "pdf"]
)


# ==========================================
# FILE PROCESSING
# ==========================================

if uploaded_file is not None:

    file_name = uploaded_file.name.lower()

    # ======================================
    # CSV PROCESSING
    # ======================================

    if file_name.endswith(".csv"):

        try:

            df = pd.read_csv(uploaded_file)

            st.success("CSV uploaded successfully!")

            # ----------------------------------
            # DISPLAY DATA
            # ----------------------------------

            st.subheader("📊 Uploaded Data")

            st.dataframe(
                df,
                use_container_width=True
            )

            st.info(
                f"Rows: {len(df)} | Columns: {len(df.columns)}"
            )

            st.divider()

            # ----------------------------------
            # CHECK AMOUNT COLUMN
            # ----------------------------------

            if "amount" not in df.columns:

                st.error(
                    "The uploaded CSV does not contain an 'amount' column."
                )

            else:

                st.subheader("🔍 Anomaly Detection")

                # Make amount numeric
                df["amount"] = pd.to_numeric(
                    df["amount"],
                    errors="coerce"
                )

                # Remove rows where amount is missing
                df = df.dropna(
                    subset=["amount"]
                ).copy()

                # ----------------------------------
                # CALCULATE STATISTICS
                # ----------------------------------

                mean_amount = df["amount"].mean()
                std_amount = df["amount"].std()

                # ----------------------------------
                # CALCULATE ANOMALY SCORE
                # ----------------------------------

                if std_amount == 0 or pd.isna(std_amount):

                    df["anomaly_score"] = 0.0

                else:

                    df["anomaly_score"] = (
                        (
                            (df["amount"] - mean_amount).abs()
                            / std_amount
                        ) * 25
                    )

                    df["anomaly_score"] = df[
                        "anomaly_score"
                    ].clip(0, 100)

                # ----------------------------------
                # RISK CLASSIFICATION
                # ----------------------------------

                def risk_level(score):

                    if score >= 75:
                        return "CRITICAL"

                    elif score >= 50:
                        return "HIGH"

                    elif score >= 25:
                        return "MEDIUM"

                    else:
                        return "LOW"

                df["risk_level"] = df[
                    "anomaly_score"
                ].apply(risk_level)

                # ----------------------------------
                # DASHBOARD METRICS
                # ----------------------------------

                total_transactions = len(df)

                high_risk_count = len(
                    df[
                        df["risk_level"].isin(
                            ["HIGH", "CRITICAL"]
                        )
                    ]
                )

                medium_risk_count = len(
                    df[
                        df["risk_level"] == "MEDIUM"
                    ]
                )

                low_risk_count = len(
                    df[
                        df["risk_level"] == "LOW"
                    ]
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Total Transactions",
                    total_transactions
                )

                col2.metric(
                    "🔴 High/Critical",
                    high_risk_count
                )

                col3.metric(
                    "🟠 Medium Risk",
                    medium_risk_count
                )

                col4.metric(
                    "🟢 Low Risk",
                    low_risk_count
                )

                st.divider()

                # ----------------------------------
                # ALL ANALYZED TRANSACTIONS
                # ----------------------------------

                st.subheader(
                    "📋 Anomaly Analysis"
                )

                st.dataframe(
                    df,
                    use_container_width=True
                )

                # ----------------------------------
                # HIGH-RISK TRANSACTIONS
                # ----------------------------------

                st.subheader(
                    "⚠️ High-Risk Transactions"
                )

                high_risk = df[
                    df["risk_level"].isin(
                        ["HIGH", "CRITICAL"]
                    )
                ].sort_values(
                    "anomaly_score",
                    ascending=False
                )

                if len(high_risk) > 0:

                    display_columns = [
                        "transaction_id",
                        "order_id",
                        "customer_id",
                        "amount",
                        "payment_date",
                        "payment_status",
                        "anomaly_score",
                        "risk_level"
                    ]

                    # Only use columns that exist
                    display_columns = [
                        col
                        for col in display_columns
                        if col in high_risk.columns
                    ]

                    st.dataframe(
                        high_risk[display_columns],
                        use_container_width=True
                    )

                    st.warning(
                        f"{len(high_risk)} high/critical-risk "
                        "transactions detected."
                    )

                else:

                    st.success(
                        "No high-risk transactions detected."
                    )

                # ----------------------------------
                # DOWNLOAD RESULTS
                # ----------------------------------

                st.subheader(
                    "📥 Download Analysis"
                )

                csv_output = df.to_csv(
                    index=False
                )

                st.download_button(
                    label="Download Anomaly Results",
                    data=csv_output,
                    file_name="LedgerLens_Anomaly_Results.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(
                f"Error reading CSV: {e}"
            )


    # ======================================
    # PDF PROCESSING
    # ======================================

    elif file_name.endswith(".pdf"):

        st.success(
            "PDF uploaded successfully!"
        )

        st.subheader(
            "📄 PDF Information"
        )

        st.write(
            f"File name: {uploaded_file.name}"
        )

        # ----------------------------------
        # EXTRACT PDF TEXT
        # ----------------------------------

        try:

            reader = PdfReader(
                uploaded_file
            )

            extracted_text = ""

            for page in reader.pages:

                text = page.extract_text()

                if text:

                    extracted_text += (
                        text + "\n"
                    )

            # ----------------------------------
            # TEXT EXTRACTION RESULT
            # ----------------------------------

            if extracted_text.strip():

                st.success(
                    "PDF text extracted successfully!"
                )

                st.subheader(
                    "📑 Extracted Financial Data"
                )

                st.text_area(
                    "PDF Content",
                    extracted_text,
                    height=400
                )

                st.divider()

                # ----------------------------------
                # AI INVESTIGATION
                # ----------------------------------

                st.subheader(
                    "🤖 AI Financial Investigation"
                )

                if client is None:

                    st.error(
                        "Gemini API key is not configured. "
                        "Please check Streamlit Secrets."
                    )

                else:

                    if st.button(
                        "🔎 Investigate Financial Data"
                    ):

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

                        with st.spinner(
                            "AI is investigating the financial data..."
                        ):

                            try:

                                response = client.models.generate_content(
                                    model="gemini-3.5-flash",
                                    contents=investigation_prompt
                                )

                                st.success(
                                    "Investigation completed!"
                                )

                                st.subheader(
                                    "📋 Investigation Report"
                                )

                                st.text_area(
                                    "LedgerLens AI Report",
                                    response.text,
                                    height=500
                                )

                            except Exception as e:

                                st.error(
                                    f"AI investigation failed: {e}"
                                )

            else:

                st.warning(
                    "No readable text was found in this PDF."
                )

        except Exception as e:

            st.error(
                f"Error extracting PDF text: {e}"
            )
