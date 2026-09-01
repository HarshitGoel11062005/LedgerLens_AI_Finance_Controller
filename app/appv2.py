import streamlit as st
import pandas as pd
import io
import re
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
    "Upload financial data and LedgerLens will automatically "
    "process the available information, detect anomalies, "
    "identify financial risks, and assist with investigation."
)

st.divider()


# =========================================================
# DATASET TYPE DETECTION
# =========================================================

def detect_dataset_type(df):

    columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]

    column_text = " ".join(columns)

    # Transaction data
    if any(
        keyword in column_text
        for keyword in [
            "transaction_id",
            "payment_id",
            "order_id"
        ]
    ):
        return "Transactions"

    # Refund data
    if any(
        keyword in column_text
        for keyword in [
            "refund_id",
            "refund_amount",
            "refund_date"
        ]
    ):
        return "Refunds"

    # Fee data
    if any(
        keyword in column_text
        for keyword in [
            "fee",
            "fee_amount",
            "processing_fee"
        ]
    ):
        return "Fees"

    # Settlement data
    if any(
        keyword in column_text
        for keyword in [
            "settlement_id",
            "settlement_amount",
            "settlement_date"
        ]
    ):
        return "Settlements"

    # Generic financial data
    if "amount" in columns:
        return "Financial Data"

    return "Unknown Financial Data"


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    extracted_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            extracted_text += text + "\n"

    return extracted_text


# =========================================================
# PDF TABLE DETECTION
# =========================================================

def pdf_text_to_dataframe(text):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        return None

    # Try pipe-separated tables
    pipe_lines = [
        line for line in lines
        if "|" in line
    ]

    if len(pipe_lines) >= 2:

        rows = []

        for line in pipe_lines:

            parts = [
                part.strip()
                for part in line.split("|")
            ]

            if len(parts) >= 2:
                rows.append(parts)

        if len(rows) >= 2:

            header = rows[0]

            # Remove markdown separator rows
            if all(
                re.fullmatch(r"[-: ]+", item or "")
                for item in rows[1]
            ):
                rows = rows[2:]

            else:
                rows = rows[1:]

            if rows:

                width = len(header)

                cleaned_rows = [
                    row[:width] + [""] * max(
                        0,
                        width - len(row)
                    )
                    for row in rows
                ]

                return pd.DataFrame(
                    cleaned_rows,
                    columns=header
                )

    # Try comma-separated text
    comma_lines = [
        line for line in lines
        if "," in line
    ]

    if len(comma_lines) >= 2:

        try:

            csv_text = "\n".join(
                comma_lines
            )

            df = pd.read_csv(
                io.StringIO(csv_text)
            )

            if len(df.columns) >= 2:
                return df

        except Exception:
            pass

    # Try whitespace-separated financial table
    whitespace_rows = []

    for line in lines:

        parts = re.split(
            r"\s{2,}",
            line
        )

        if len(parts) >= 2:
            whitespace_rows.append(parts)

    if len(whitespace_rows) >= 3:

        header = whitespace_rows[0]

        data_rows = whitespace_rows[1:]

        width = len(header)

        cleaned_rows = [
            row[:width] + [""] * max(
                0,
                width - len(row)
            )
            for row in data_rows
        ]

        return pd.DataFrame(
            cleaned_rows,
            columns=header
        )

    return None


# =========================================================
# NORMALIZE COLUMN NAMES
# =========================================================

def normalize_columns(df):

    df = df.copy()

    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in df.columns
    ]

    return df


# =========================================================
# ANOMALY ANALYSIS
# =========================================================

def analyze_financial_data(df):

    df = normalize_columns(df)

    # Find amount-like column
    amount_column = None

    possible_amount_columns = [
        "amount",
        "transaction_amount",
        "payment_amount",
        "refund_amount",
        "fee_amount",
        "settlement_amount",
        "total_amount",
        "value"
    ]

    for column in possible_amount_columns:

        if column in df.columns:
            amount_column = column
            break

    if amount_column is None:

        # Try to find a column containing amount/value
        for column in df.columns:

            if (
                "amount" in column
                or "value" in column
            ):
                amount_column = column
                break

    if amount_column is None:

        return None, (
            "No financial amount column could be identified."
        )

    # Convert amount
    df[amount_column] = pd.to_numeric(
        df[amount_column],
        errors="coerce"
    )

    valid_df = df.dropna(
        subset=[amount_column]
    ).copy()

    if len(valid_df) == 0:

        return None, (
            "No valid financial amounts were found."
        )

    mean_amount = valid_df[
        amount_column
    ].mean()

    std_amount = valid_df[
        amount_column
    ].std()

    # Calculate anomaly score
    if std_amount == 0 or pd.isna(std_amount):

        valid_df["anomaly_score"] = 0.0

    else:

        valid_df["anomaly_score"] = (
            (
                (
                    valid_df[amount_column]
                    - mean_amount
                ).abs()
                / std_amount
            ) * 25
        )

        valid_df["anomaly_score"] = valid_df[
            "anomaly_score"
        ].clip(0, 100)

    # Risk classification
    def risk_level(score):

        if score >= 75:
            return "CRITICAL"

        elif score >= 50:
            return "HIGH"

        elif score >= 25:
            return "MEDIUM"

        return "LOW"

    valid_df["risk_level"] = valid_df[
        "anomaly_score"
    ].apply(risk_level)

    return valid_df, None


# =========================================================
# DISPLAY DASHBOARD
# =========================================================

def display_dashboard(df, dataset_name):

    st.divider()

    st.header(
        f"📊 Financial Risk Dashboard"
    )

    st.caption(
        f"Dataset detected as: {dataset_name}"
    )

    analyzed_df, error_message = analyze_financial_data(
        df
    )

    if error_message:

        st.warning(error_message)

        st.subheader("📋 Uploaded Data")

        st.dataframe(
            df,
            use_container_width=True
        )

        return

    # =====================================================
    # METRICS
    # =====================================================

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
        "Total Records",
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

    # =====================================================
    # ANALYZED DATA
    # =====================================================

    st.divider()

    st.subheader(
        "📋 Financial Data Analysis"
    )

    st.dataframe(
        analyzed_df,
        use_container_width=True
    )

    # =====================================================
    # HIGH RISK
    # =====================================================

    st.divider()

    st.subheader(
        "⚠️ High-Risk Financial Records"
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

        st.dataframe(
            high_risk,
            use_container_width=True
        )

        st.warning(
            f"{len(high_risk)} high/critical-risk "
            "records detected."
        )

    else:

        st.success(
            "No high-risk records detected."
        )

    # =====================================================
    # DOWNLOAD CSV
    # =====================================================

    st.divider()

    st.subheader(
        "📥 Export Processed Data"
    )

    csv_data = analyzed_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Processed CSV",
        data=csv_data,
        file_name="LedgerLens_Processed_Data.csv",
        mime="text/csv"
    )


# =========================================================
# UPLOAD MULTIPLE FILES
# =========================================================

st.header(
    "📂 Upload Financial Data"
)

st.write(
    "Upload one or more CSV or PDF financial files. "
    "LedgerLens will automatically inspect the data."
)

uploaded_files = st.file_uploader(
    "Choose financial files",
    type=["csv", "pdf"],
    accept_multiple_files=True
)


# =========================================================
# PROCESS UPLOADED FILES
# =========================================================

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} file(s) uploaded successfully."
    )

    for uploaded_file in uploaded_files:

        st.divider()

        st.subheader(
            f"📄 {uploaded_file.name}"
        )

        file_name = uploaded_file.name.lower()

        # =================================================
        # CSV
        # =================================================

        if file_name.endswith(".csv"):

            try:

                df = pd.read_csv(
                    uploaded_file
                )

                dataset_type = detect_dataset_type(
                    df
                )

                st.success(
                    f"Detected data type: {dataset_type}"
                )

                display_dashboard(
                    df,
                    dataset_type
                )

            except Exception as e:

                st.error(
                    f"Error processing CSV: {e}"
                )

        # =================================================
        # PDF
        # =================================================

        elif file_name.endswith(".pdf"):

            try:

                extracted_text = extract_pdf_text(
                    uploaded_file
                )

                if not extracted_text.strip():

                    st.warning(
                        "No readable text was found in this PDF."
                    )

                    continue

                st.success(
                    "PDF text extracted successfully."
                )

                # -----------------------------------------
                # Try to detect a table
                # -----------------------------------------

                pdf_df = pdf_text_to_dataframe(
                    extracted_text
                )

                if pdf_df is not None:

                    pdf_df = normalize_columns(
                        pdf_df
                    )

                    dataset_type = detect_dataset_type(
                        pdf_df
                    )

                    st.success(
                        "Structured financial data detected "
                        f"({dataset_type})."
                    )

                    st.subheader(
                        "📊 Extracted Table"
                    )

                    st.dataframe(
                        pdf_df,
                        use_container_width=True
                    )

                    # Dashboard
                    display_dashboard(
                        pdf_df,
                        dataset_type
                    )

                    # Download converted CSV
                    converted_csv = pdf_df.to_csv(
                        index=False
                    )

                    st.download_button(
                        label="⬇️ Download Converted CSV",
                        data=converted_csv,
                        file_name=(
                            uploaded_file.name
                            .replace(".pdf", "")
                            + "_converted.csv"
                        ),
                        mime="text/csv"
                    )

                else:

                    # -------------------------------------
                    # No table detected
                    # -------------------------------------

                    st.info(
                        "No structured transaction table was "
                        "detected in this PDF."
                    )

                    st.subheader(
                        "📑 Extracted Financial Report"
                    )

                    st.text_area(
                        "PDF Content",
                        extracted_text,
                        height=350
                    )

                    # -------------------------------------
                    # Gemini investigation
                    # -------------------------------------

                    st.subheader(
                        "🤖 AI Financial Investigation"
                    )

                    if not gemini_available:

                        st.error(
                            "Gemini API is not configured. "
                            "Check GEMINI_API_KEY in Streamlit Secrets."
                        )

                    else:

                        investigate = st.button(
                            "🔎 Investigate PDF",
                            key=(
                                "investigate_"
                                + uploaded_file.name
                            )
                        )

                        if investigate:

                            prompt = f"""
You are LedgerLens, an AI financial investigation assistant.

Analyze ONLY the financial evidence provided below.

Rules:
1. Identify suspicious or abnormal financial activity.
2. Explain the evidence clearly.
3. Mention financial impact only when supported by the evidence.
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
                                "AI is investigating the financial report..."
                            ):

                                try:

                                    response = (
                                        client.models.generate_content(
                                            model="gemini-3.5-flash",
                                            contents=prompt
                                        )
                                    )

                                    st.success(
                                        "Investigation completed!"
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

            except Exception as e:

                st.error(
                    f"Error processing PDF: {e}"
                )


# =========================================================
# NO FILE MESSAGE
# =========================================================

else:

    st.info(
        "Upload your financial CSV/PDF files above "
        "to generate the LedgerLens dashboard."
    )

    st.caption(
        "Tip: You can upload multiple financial files at once."
    )
