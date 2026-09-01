import streamlit as st
import pandas as pd
from pypdf import PdfReader
from google import genai


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LedgerLens",
    page_icon="💰",
    layout="wide"
)


# =========================================================
# GEMINI CLIENT
# =========================================================

try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )
    gemini_available = True

except Exception:
    client = None
    gemini_available = False


# =========================================================
# HEADER
# =========================================================

st.title("💰 LedgerLens")
st.subheader("AI Finance Controller")

st.write(
    "Detect financial anomalies, investigate suspicious transactions, "
    "and generate AI-powered financial explanations."
)

st.divider()


# =========================================================
# FUNCTION: ANOMALY DETECTION
# =========================================================

def analyze_transactions(df):

    df = df.copy()

    # Check amount column
    if "amount" not in df.columns:
        return None, "The CSV must contain an 'amount' column."

    # Convert amount to numeric
    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    # Remove invalid amounts
    df = df.dropna(
        subset=["amount"]
    ).copy()

    if len(df) == 0:
        return None, "No valid transaction amounts were found."

    # Calculate statistics
    mean_amount = df["amount"].mean()
    std_amount = df["amount"].std()

    # Calculate anomaly score
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

    # Risk classification
    def get_risk_level(score):

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
    ].apply(get_risk_level)

    return df, None


# =========================================================
# AUTOMATIC DEMO DATA
# =========================================================

st.header("📊 Financial Risk Dashboard")

st.write(
    "LedgerLens automatically loads the project's transaction "
    "dataset for demonstration."
)


try:

    default_df = pd.read_csv(
        "data/transactions.csv"
    )

    st.success(
        "Demo transaction data loaded successfully."
    )

except Exception as e:

    default_df = None

    st.error(
        "Could not load the default transaction data."
    )

    st.caption(
        f"Technical details: {e}"
    )


# =========================================================
# OPTIONAL USER UPLOAD
# =========================================================

st.subheader("📂 Analyze Your Own Financial Data")

st.write(
    "You can optionally upload your own CSV or PDF. "
    "If you do not upload anything, LedgerLens uses the "
    "built-in demo transaction data."
)

uploaded_file = st.file_uploader(
    "Upload CSV or PDF",
    type=["csv", "pdf"]
)


# =========================================================
# DETERMINE ACTIVE DATA
# =========================================================

active_df = None


if uploaded_file is not None:

    file_name = uploaded_file.name.lower()

    # =====================================================
    # USER CSV
    # =====================================================

    if file_name.endswith(".csv"):

        try:

            active_df = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"Using uploaded file: {uploaded_file.name}"
            )

        except Exception as e:

            st.error(
                f"Error reading uploaded CSV: {e}"
            )


    # =====================================================
    # USER PDF
    # =====================================================

    elif file_name.endswith(".pdf"):

        st.success(
            f"PDF uploaded: {uploaded_file.name}"
        )

        st.subheader(
            "📄 PDF Information"
        )

        st.write(
            f"File name: {uploaded_file.name}"
        )

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

            # ---------------------------------------------
            # PDF TEXT
            # ---------------------------------------------

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
                    height=350
                )

                st.divider()

                # -----------------------------------------
                # GEMINI INVESTIGATION
                # -----------------------------------------

                st.subheader(
                    "🤖 AI Financial Investigation"
                )

                if not gemini_available:

                    st.error(
                        "Gemini API is not configured. "
                        "Please check GEMINI_API_KEY in "
                        "Streamlit Secrets."
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


# =========================================================
# USE DEFAULT DATA IF NO CSV WAS UPLOADED
# =========================================================

if active_df is None and uploaded_file is None:

    active_df = default_df


# =========================================================
# TRANSACTION DASHBOARD
# =========================================================

if active_df is not None:

    analyzed_df, error_message = analyze_transactions(
        active_df
    )

    if error_message:

        st.error(
            error_message
        )

    elif analyzed_df is not None:

        st.divider()

        st.subheader(
            "🔍 Anomaly Detection"
        )

        # =================================================
        # DASHBOARD METRICS
        # =================================================

        total_transactions = len(
            analyzed_df
        )

        high_risk_count = len(
            analyzed_df[
                analyzed_df["risk_level"].isin(
                    ["HIGH", "CRITICAL"]
                )
            ]
        )

        medium_risk_count = len(
            analyzed_df[
                analyzed_df["risk_level"] == "MEDIUM"
            ]
        )

        low_risk_count = len(
            analyzed_df[
                analyzed_df["risk_level"] == "LOW"
            ]
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Transactions",
            total_transactions
        )

        col2.metric(
            "🔴 High / Critical",
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

        # =================================================
        # DATA PREVIEW
        # =================================================

        st.divider()

        st.subheader(
            "📋 Transaction Analysis"
        )

        st.dataframe(
            analyzed_df,
            use_container_width=True
        )

        # =================================================
        # HIGH RISK TRANSACTIONS
        # =================================================

        st.divider()

        st.subheader(
            "⚠️ High-Risk Transactions"
        )

        high_risk = analyzed_df[
            analyzed_df["risk_level"].isin(
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

            # Only display columns that exist
            display_columns = [
                column
                for column in display_columns
                if column in high_risk.columns
            ]

            st.dataframe(
                high_risk[
                    display_columns
                ],
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

        # =================================================
        # DOWNLOAD ANALYSIS
        # =================================================

        st.divider()

        st.subheader(
            "📥 Export Analysis"
        )

        analysis_csv = analyzed_df.to_csv(
            index=False
        )

        st.download_button(
            label="Download Anomaly Results",
            data=analysis_csv,
            file_name="LedgerLens_Anomaly_Results.csv",
            mime="text/csv"
        )
