import os
import io
import re
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from google import genai


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LedgerLens | AI Finance Controller",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GEMINI CLIENT
# ============================================================

try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )
    AI_CONNECTED = True
except Exception:
    client = None
    AI_CONNECTED = False


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #101827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .main-subtitle {
        font-size: 18px;
        color: #64748b;
        margin-top: 0px;
    }

    /* Section spacing */
    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Status box */
    .status-box {
        padding: 14px;
        border-radius: 12px;
        background-color: #123333;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 💰 LedgerLens")

    st.caption("AI Finance Controller")

    st.divider()

    if AI_CONNECTED:
        st.success("🤖 AI Engine Connected")
    else:
        st.warning("⚠️ AI Engine Not Connected")

    st.divider()

    st.markdown("### ⚙️ Dashboard Controls")

    risk_filter = st.multiselect(
        "Risk levels to display",
        ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        default=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    )

    st.divider()

    st.markdown("### 📊 About LedgerLens")

    st.caption(
        "LedgerLens detects financial anomalies, "
        "classifies transaction risk, and provides "
        "AI-powered financial investigation."
    )

    st.caption("Version 3.0")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💰 LedgerLens</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'AI-powered financial risk intelligence and investigation platform'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def detect_dataset_type(df):

    columns = [str(c).lower() for c in df.columns]

    if "transaction_id" in columns:
        return "Transactions"

    if "refund_id" in columns:
        return "Refunds"

    if "fee_id" in columns:
        return "Fees"

    if "settlement_id" in columns:
        return "Settlements"

    if "amount" in columns or "amount_inr" in columns:
        return "Financial Records"

    return "Financial Dataset"


def find_amount_column(df):

    possible_columns = [
        "amount",
        "amount_inr",
        "transaction_amount",
        "value",
        "total_amount",
        "refund_amount",
        "fee_amount",
        "settlement_amount"
    ]

    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for column in possible_columns:
        if column in lower_map:
            return lower_map[column]

    # Try to find any numeric column
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:
        return numeric_columns[0]

    return None


def calculate_anomalies(df):

    df = df.copy()

    amount_column = find_amount_column(df)

    if amount_column is None:
        return df, None

    df[amount_column] = pd.to_numeric(
        df[amount_column],
        errors="coerce"
    )

    mean_amount = df[amount_column].mean()
    std_amount = df[amount_column].std()

    if pd.isna(std_amount) or std_amount == 0:

        df["anomaly_score"] = 0.0

    else:

        df["anomaly_score"] = (
            (
                (df[amount_column] - mean_amount).abs()
                / std_amount
            ) * 25
        )

        df["anomaly_score"] = (
            df["anomaly_score"]
            .fillna(0)
            .clip(0, 100)
        )

    def risk_level(score):

        if score >= 75:
            return "CRITICAL"

        elif score >= 50:
            return "HIGH"

        elif score >= 25:
            return "MEDIUM"

        return "LOW"

    df["risk_level"] = df["anomaly_score"].apply(
        risk_level
    )

    return df, amount_column


def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    extracted_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            extracted_text += text + "\n"

    return extracted_text


def format_amount(value):

    try:
        return f"₹{float(value):,.2f}"
    except Exception:
        return str(value)


def ai_investigation(evidence, investigation_type="full"):

    if not AI_CONNECTED:
        return "Gemini API is not connected. Please check GEMINI_API_KEY in Streamlit Secrets."

    if investigation_type == "single":

        prompt = f"""
You are LedgerLens, an AI financial investigation assistant.

Investigate ONLY the transaction evidence provided below.

Do not invent information.

Analyze:
1. Why this transaction may be suspicious.
2. Its anomaly/risk indicators.
3. Financial impact.
4. Possible explanation.
5. Recommended human review action.

Return exactly:

Finding:
Evidence:
Financial Impact:
Possible Cause:
Recommendation:
Confidence:

TRANSACTION EVIDENCE:
{evidence}
"""

    else:

        prompt = f"""
You are LedgerLens, an AI financial investigation assistant.

Analyze ONLY the financial evidence provided below.

Rules:
1. Identify suspicious or abnormal financial activity.
2. Explain the evidence clearly.
3. Identify important risk patterns.
4. Calculate or mention financial impact only when supported.
5. Do not invent missing information.
6. If evidence is insufficient, explicitly say so.
7. Do not authorize or execute financial transactions.
8. Recommendations are for human review only.
9. Be concise and professional.

Return exactly:

Executive Summary:
Key Findings:
High-Risk Activity:
Financial Impact:
Possible Causes:
Recommended Actions:
Confidence:

FINANCIAL EVIDENCE:
{evidence}
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"AI investigation failed: {e}"


# ============================================================
# DATA SOURCE
# ============================================================

st.markdown(
    '<div class="section-title">📂 Financial Data</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a CSV or PDF financial report",
    type=["csv", "pdf"],
    help="Upload your own financial data. LedgerLens will analyze it automatically."
)


# ============================================================
# DATA LOADING
# ============================================================

df = None
pdf_text = None
source_name = None


# ------------------------------------------------------------
# USER UPLOAD
# ------------------------------------------------------------

if uploaded_file is not None:

    source_name = uploaded_file.name

    if uploaded_file.name.lower().endswith(".csv"):

        try:

            df = pd.read_csv(uploaded_file)

            st.success(
                f"✅ {uploaded_file.name} loaded successfully."
            )

        except Exception as e:

            st.error(
                f"Could not read CSV: {e}"
            )

    elif uploaded_file.name.lower().endswith(".pdf"):

        try:

            pdf_text = extract_pdf_text(uploaded_file)

            st.success(
                f"✅ {uploaded_file.name} loaded successfully."
            )

        except Exception as e:

            st.error(
                f"Could not read PDF: {e}"
            )


# ------------------------------------------------------------
# AUTOMATIC DEMO DATA
# ------------------------------------------------------------

else:

    default_path = "data/transactions.csv"

    if os.path.exists(default_path):

        try:

            df = pd.read_csv(default_path)

            source_name = "Demo transactions.csv"

            st.info(
                "📊 Demo financial data loaded automatically. "
                "Upload your own CSV or PDF above to analyze new data."
            )

        except Exception as e:

            st.error(
                f"Could not load demo dataset: {e}"
            )


# ============================================================
# PDF MODE
# ============================================================

if pdf_text is not None:

    st.markdown(
        '<div class="section-title">📄 PDF Financial Report</div>',
        unsafe_allow_html=True
    )

    st.info(
        "LedgerLens successfully extracted text from the PDF."
    )

    with st.expander(
        "View extracted PDF data"
    ):

        st.text_area(
            "Extracted Content",
            pdf_text,
            height=350
        )

    st.divider()

    st.markdown(
        '<div class="section-title">🤖 AI Financial Investigation</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "🚀 Run AI Investigation",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "LedgerLens AI is investigating the financial data..."
        ):

            report = ai_investigation(
                pdf_text,
                "full"
            )

        st.success(
            "AI investigation completed."
        )

        st.text_area(
            "LedgerLens AI Report",
            report,
            height=500
        )


# ============================================================
# CSV DASHBOARD
# ============================================================

if df is not None:

    dataset_type = detect_dataset_type(df)

    # --------------------------------------------------------
    # ANOMALY DETECTION
    # --------------------------------------------------------

    analyzed_df, amount_column = calculate_anomalies(df)

    if amount_column is None:

        st.error(
            "No financial amount column was detected in this dataset."
        )

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        # ====================================================
        # DATASET HEADER
        # ====================================================

        st.markdown(
            f"### 📊 Financial Risk Dashboard"
        )

        st.caption(
            f"Dataset detected: **{dataset_type}**  |  "
            f"Source: **{source_name}**"
        )

        # ====================================================
        # KPI SECTION
        # ====================================================

        total_records = len(analyzed_df)

        critical_count = len(
            analyzed_df[
                analyzed_df["risk_level"] == "CRITICAL"
            ]
        )

        high_count = len(
            analyzed_df[
                analyzed_df["risk_level"] == "HIGH"
            ]
        )

        medium_count = len(
            analyzed_df[
                analyzed_df["risk_level"] == "MEDIUM"
            ]
        )

        low_count = len(
            analyzed_df[
                analyzed_df["risk_level"] == "LOW"
            ]
        )

        high_critical_count = (
            critical_count + high_count
        )

        total_amount = analyzed_df[
            amount_column
        ].sum()

        # ----------------------------------------------------
        # KPI CARDS
        # ----------------------------------------------------

        st.markdown("### 📌 Risk Overview")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Total Records",
            f"{total_records:,}"
        )

        c2.metric(
            "🔴 High / Critical",
            f"{high_critical_count:,}"
        )

        c3.metric(
            "🟠 Medium Risk",
            f"{medium_count:,}"
        )

        c4.metric(
            "🟢 Low Risk",
            f"{low_count:,}"
        )

        c5.metric(
            "💰 Total Amount",
            format_amount(total_amount)
        )

        st.divider()

        # ====================================================
        # CHARTS
        # ====================================================

        st.markdown("### 📈 Financial Analytics")

        chart1, chart2 = st.columns(2)

        # ----------------------------------------------------
        # RISK DISTRIBUTION
        # ----------------------------------------------------

        with chart1:

            st.markdown("#### 🎯 Risk Distribution")

            risk_data = pd.DataFrame(
                {
                    "Risk Level": [
                        "LOW",
                        "MEDIUM",
                        "HIGH",
                        "CRITICAL"
                    ],
                    "Transactions": [
                        low_count,
                        medium_count,
                        high_count,
                        critical_count
                    ]
                }
            )

            st.bar_chart(
                risk_data.set_index("Risk Level")
            )

        # ----------------------------------------------------
        # AMOUNT BY RISK
        # ----------------------------------------------------

        with chart2:

            st.markdown(
                "#### 💰 Financial Exposure by Risk"
            )

            exposure = (
                analyzed_df
                .groupby("risk_level")[amount_column]
                .sum()
                .reindex(
                    [
                        "LOW",
                        "MEDIUM",
                        "HIGH",
                        "CRITICAL"
                    ],
                    fill_value=0
                )
            )

            exposure_df = pd.DataFrame(
                {
                    "Risk Level": exposure.index,
                    "Amount": exposure.values
                }
            )

            st.bar_chart(
                exposure_df.set_index("Risk Level")
            )

        # ====================================================
        # ADDITIONAL CHART
        # ====================================================

        if len(analyzed_df) > 1:

            st.markdown(
                "#### 📊 Transaction Amount Trend"
            )

            trend_df = analyzed_df[
                [amount_column]
            ].reset_index(drop=True)

            trend_df.columns = ["Amount"]

            st.line_chart(
                trend_df
            )

        st.divider()

        # ====================================================
        # FILTER DATA
        # ====================================================

        st.markdown(
            "### 🔎 Financial Data Analysis"
        )

        filtered_df = analyzed_df[
            analyzed_df["risk_level"].isin(
                risk_filter
            )
        ]

        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=420
        )

        # ====================================================
        # HIGH RISK TRANSACTIONS
        # ====================================================

        st.markdown(
            "### ⚠️ High-Risk Transactions"
        )

        high_risk_df = analyzed_df[
            analyzed_df["risk_level"].isin(
                ["HIGH", "CRITICAL"]
            )
        ].sort_values(
            "anomaly_score",
            ascending=False
        )

        if len(high_risk_df) > 0:

            st.dataframe(
                high_risk_df,
                use_container_width=True,
                height=350
            )

        else:

            st.success(
                "No HIGH or CRITICAL transactions detected."
            )

        # ====================================================
        # SINGLE TRANSACTION INVESTIGATION
        # ====================================================

        st.divider()

        st.markdown(
            "### 🔍 Single Transaction Investigation"
        )

        st.write(
            "Select an individual transaction to investigate "
            "its financial risk using LedgerLens AI."
        )

        # Create readable transaction labels

        transaction_labels = []

        for index, row in analyzed_df.iterrows():

            if "transaction_id" in analyzed_df.columns:

                identifier = str(
                    row["transaction_id"]
                )

            elif "id" in analyzed_df.columns:

                identifier = str(
                    row["id"]
                )

            else:

                identifier = f"Record {index + 1}"

            risk = row["risk_level"]

            score = row["anomaly_score"]

            transaction_labels.append(
                f"{identifier} | {risk} | Score: {score:.2f}"
            )

        selected_transaction = st.selectbox(
            "Select a transaction",
            transaction_labels
        )

        selected_position = transaction_labels.index(
            selected_transaction
        )

        selected_row = analyzed_df.iloc[
            selected_position
        ]

        # ----------------------------------------------------
        # DISPLAY TRANSACTION
        # ----------------------------------------------------

        st.markdown(
            "#### 📋 Transaction Details"
        )

        transaction_display = pd.DataFrame(
            selected_row
        ).reset_index()

        transaction_display.columns = [
            "Field",
            "Value"
        ]

        st.dataframe(
            transaction_display,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # AI SINGLE INVESTIGATION
        # ----------------------------------------------------

        if st.button(
            "🔎 Investigate This Transaction",
            type="primary"
        ):

            evidence = selected_row.to_string()

            with st.spinner(
                "AI is investigating this transaction..."
            ):

                report = ai_investigation(
                    evidence,
                    "single"
                )

            st.success(
                "Transaction investigation completed."
            )

            st.text_area(
                "AI Transaction Investigation",
                report,
                height=400
            )

        # ====================================================
        # FULL AI INVESTIGATION
        # ====================================================

        st.divider()

        st.markdown(
            "### 🤖 AI Financial Investigation"
        )

        st.write(
            "Run a complete AI investigation across the "
            "uploaded financial dataset."
        )

        if st.button(
            "🚀 Run Full AI Investigation",
            type="primary",
            use_container_width=True
        ):

            # Limit extremely large datasets
            # so API requests remain manageable

            ai_data = analyzed_df.copy()

            if len(ai_data) > 200:

                ai_data = ai_data.head(200)

            evidence = ai_data.to_string(
                index=False
            )

            with st.spinner(
                "LedgerLens AI is investigating the financial data..."
            ):

                report = ai_investigation(
                    evidence,
                    "full"
                )

            st.success(
                "Full AI investigation completed."
            )

            st.text_area(
                "LedgerLens AI Investigation Report",
                report,
                height=550
            )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.divider()

        st.markdown(
            "### 📥 Export Results"
        )

        csv_buffer = io.StringIO()

        analyzed_df.to_csv(
            csv_buffer,
            index=False
        )

        st.download_button(
            label="⬇️ Download Analyzed CSV",
            data=csv_buffer.getvalue(),
            file_name="ledgerlens_analyzed_data.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "LedgerLens • AI Finance Controller • "
    "AI recommendations are for human review only."
)
