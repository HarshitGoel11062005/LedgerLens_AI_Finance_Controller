import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import re
from pypdf import PdfReader
from google import genai

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LedgerLens | AI Finance Controller",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# Only styling - NO HTML DIV CARDS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
}

h2 {
    font-weight: 750 !important;
}

h3 {
    font-weight: 700 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e5e7eb;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
}

/* Buttons */
.stButton > button {
    border-radius: 9px;
    font-weight: 600;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: white;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# GEMINI CLIENT
# ============================================================

try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )
    gemini_available = True

except Exception:
    client = None
    gemini_available = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 💰 LedgerLens")

    st.caption("AI Finance Controller")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🔎 Transaction Explorer",
            "🤖 AI Investigation",
            "📁 Data Management"
        ]
    )

    st.divider()

    st.markdown("### System Status")

    if gemini_available:
        st.success("AI Engine Connected")
    else:
        st.warning("AI Engine Not Connected")

    st.caption("LedgerLens v2.0")


# ============================================================
# HEADER
# ============================================================

st.title("LedgerLens")

st.subheader("AI Finance Controller")

st.write(
    "AI-powered financial risk intelligence for detecting anomalies, "
    "investigating suspicious transactions, and supporting human review."
)

st.divider()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):

    lower_map = {
        str(col).lower().strip(): col
        for col in df.columns
    }

    for name in possible_names:

        if name.lower() in lower_map:
            return lower_map[name.lower()]

    return None


def standardize_dataframe(df):

    df = df.copy()

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Clean column names
    df.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]

    # Detect amount column
    amount_col = find_column(
        df,
        [
            "amount",
            "amount_inr",
            "transaction_amount",
            "value",
            "total_amount",
            "price"
        ]
    )

    if amount_col:

        df["amount"] = (
            df[amount_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("$", "", regex=False)
        )

        df["amount"] = pd.to_numeric(
            df["amount"],
            errors="coerce"
        )

    # Detect date
    date_col = find_column(
        df,
        [
            "date",
            "payment_date",
            "transaction_date",
            "settlement_date",
            "refund_date"
        ]
    )

    if date_col:

        df["date"] = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )

    return df


def detect_dataset_type(df):

    columns = set(
        str(col).lower()
        for col in df.columns
    )

    if "transaction_id" in columns:
        return "Transactions"

    if "refund_id" in columns:
        return "Refunds"

    if "fee_id" in columns:
        return "Fees"

    if "settlement_id" in columns:
        return "Settlements"

    return "Financial Dataset"


def calculate_anomalies(df):

    df = df.copy()

    if "amount" not in df.columns:

        df["anomaly_score"] = 0
        df["risk_level"] = "LOW"

        return df

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    mean_amount = df["amount"].mean()
    std_amount = df["amount"].std()

    if (
        std_amount == 0
        or pd.isna(std_amount)
    ):

        df["anomaly_score"] = 0

    else:

        df["anomaly_score"] = (
            (
                (df["amount"] - mean_amount)
                .abs()
                / std_amount
            ) * 25
        )

        df["anomaly_score"] = (
            df["anomaly_score"]
            .clip(0, 100)
            .round(2)
        )

    def risk_level(score):

        if score >= 75:
            return "CRITICAL"

        elif score >= 50:
            return "HIGH"

        elif score >= 25:
            return "MEDIUM"

        return "LOW"

    df["risk_level"] = (
        df["anomaly_score"]
        .apply(risk_level)
    )

    return df


def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def text_to_dataframe(text):

    """
    Attempts to convert simple PDF table text into a dataframe.
    Works best when extracted PDF text has rows separated by lines
    and values separated by spaces/tabs.
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) < 3:
        return None

    rows = []

    for line in lines:

        # Try multiple spaces or tabs
        parts = re.split(
            r"\s{2,}|\t+",
            line
        )

        if len(parts) >= 3:
            rows.append(parts)

    if len(rows) < 3:
        return None

    try:

        header = rows[0]

        data = rows[1:]

        # Keep rows having same approximate length
        valid_rows = [
            row for row in data
            if len(row) == len(header)
        ]

        if len(valid_rows) < 2:
            return None

        df = pd.DataFrame(
            valid_rows,
            columns=header
        )

        return df

    except Exception:

        return None


def make_ai_prompt(df):

    sample = df.head(100).to_csv(
        index=False
    )

    return f"""
You are LedgerLens, an AI financial investigation assistant.

Analyze ONLY the financial evidence supplied below.

Rules:

1. Identify suspicious or abnormal financial activity.
2. Explain the evidence clearly.
3. Mention financial impact when supported by the data.
4. Do not invent information.
5. If evidence is insufficient, explicitly state that.
6. Do not authorize or execute financial transactions.
7. Recommendations are for human review only.
8. Be concise and professional.

Return exactly:

Finding:
Evidence:
Financial Impact:
Possible Cause:
Recommendation:
Confidence:

FINANCIAL DATA:

{sample}
"""


def investigate_single_transaction(row):

    transaction_text = row.to_string()

    prompt = f"""
You are LedgerLens, an AI financial investigation assistant.

Investigate ONLY this single financial record.

Do not invent information.

Explain:

Transaction:
Risk Assessment:
Evidence:
Possible Concern:
Financial Impact:
Recommended Human Review:
Confidence:

TRANSACTION:

{transaction_text}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text


# ============================================================
# LOAD DEMO DATA
# ============================================================

def load_demo_data():

    data_folder = "data"

    if not os.path.exists(data_folder):
        return None

    csv_files = [
        file
        for file in os.listdir(data_folder)
        if file.lower().endswith(".csv")
    ]

    if not csv_files:
        return None

    # Prefer transactions.csv for the main dashboard
    preferred = [
        file for file in csv_files
        if file.lower() == "transactions.csv"
    ]

    if preferred:

        path = os.path.join(
            data_folder,
            preferred[0]
        )

        try:

            df = pd.read_csv(path)

            return standardize_dataframe(df)

        except Exception:
            pass

    # Otherwise use first available CSV
    path = os.path.join(
        data_folder,
        csv_files[0]
    )

    try:

        df = pd.read_csv(path)

        return standardize_dataframe(df)

    except Exception:

        return None


# ============================================================
# SESSION STATE
# ============================================================

if "active_df" not in st.session_state:

    demo_df = load_demo_data()

    if demo_df is not None:

        st.session_state.active_df = (
            calculate_anomalies(demo_df)
        )

        st.session_state.data_source = (
            "Demo financial dataset"
        )

    else:

        st.session_state.active_df = None
        st.session_state.data_source = None


# ============================================================
# DATA MANAGEMENT
# ============================================================

if page == "📁 Data Management":

    st.header("📁 Data Management")

    st.write(
        "Upload financial data. LedgerLens automatically detects "
        "the available structure and prepares it for analysis."
    )

    uploaded_file = st.file_uploader(
        "Upload financial data",
        type=["csv", "pdf"]
    )

    if uploaded_file:

        file_name = uploaded_file.name.lower()

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        if file_name.endswith(".csv"):

            try:

                new_df = pd.read_csv(
                    uploaded_file
                )

                new_df = standardize_dataframe(
                    new_df
                )

                new_df = calculate_anomalies(
                    new_df
                )

                st.session_state.active_df = new_df

                st.session_state.data_source = (
                    uploaded_file.name
                )

                st.success(
                    "Financial CSV processed successfully."
                )

                st.dataframe(
                    new_df,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Could not process CSV: {e}"
                )

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        elif file_name.endswith(".pdf"):

            try:

                pdf_text = extract_pdf_text(
                    uploaded_file
                )

                if pdf_text.strip():

                    st.success(
                        "PDF text extracted successfully."
                    )

                    converted_df = text_to_dataframe(
                        pdf_text
                    )

                    if converted_df is not None:

                        converted_df = (
                            standardize_dataframe(
                                converted_df
                            )
                        )

                        converted_df = (
                            calculate_anomalies(
                                converted_df
                            )
                        )

                        st.session_state.active_df = (
                            converted_df
                        )

                        st.session_state.data_source = (
                            uploaded_file.name
                        )

                        st.success(
                            "PDF table converted into "
                            "an analyzable dataset."
                        )

                        st.dataframe(
                            converted_df,
                            use_container_width=True
                        )

                    else:

                        st.info(
                            "The PDF text was extracted, "
                            "but a reliable table structure "
                            "could not be detected."
                        )

                        with st.expander(
                            "View extracted PDF text"
                        ):

                            st.text(
                                pdf_text
                            )

                else:

                    st.warning(
                        "No readable text was found in the PDF."
                    )

            except Exception as e:

                st.error(
                    f"Could not process PDF: {e}"
                )


# ============================================================
# GET ACTIVE DATA
# ============================================================

df = st.session_state.active_df


# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.header("📊 Financial Risk Dashboard")

    if df is None:

        st.info(
            "No financial dataset is currently available. "
            "Upload a CSV or PDF from Data Management."
        )

        st.stop()

    dataset_type = detect_dataset_type(df)

    st.caption(
        f"Dataset: **{dataset_type}**  |  "
        f"Source: **{st.session_state.data_source}**"
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    total_records = len(df)

    high_critical = len(
        df[
            df["risk_level"].isin(
                ["HIGH", "CRITICAL"]
            )
        ]
    )

    medium = len(
        df[
            df["risk_level"] == "MEDIUM"
        ]
    )

    low = len(
        df[
            df["risk_level"] == "LOW"
        ]
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Records",
            total_records
        )

    with col2:

        st.metric(
            "🔴 High / Critical",
            high_critical
        )

    with col3:

        st.metric(
            "🟠 Medium Risk",
            medium
        )

    with col4:

        st.metric(
            "🟢 Low Risk",
            low
        )

    st.divider()

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    chart1, chart2 = st.columns(2)

    with chart1:

        st.subheader("🎯 Risk Distribution")

        risk_counts = (
            df["risk_level"]
            .value_counts()
            .reindex(
                [
                    "CRITICAL",
                    "HIGH",
                    "MEDIUM",
                    "LOW"
                ],
                fill_value=0
            )
        )

        st.bar_chart(
            risk_counts
        )

    with chart2:

        st.subheader("💰 Financial Overview")

        if "amount" in df.columns:

            financial_summary = pd.DataFrame(
                {
                    "Metric": [
                        "Total Amount",
                        "Average Amount",
                        "Maximum Amount",
                        "Minimum Amount"
                    ],
                    "Value": [
                        df["amount"].sum(),
                        df["amount"].mean(),
                        df["amount"].max(),
                        df["amount"].min()
                    ]
                }
            )

            financial_summary = (
                financial_summary
                .set_index("Metric")
            )

            st.bar_chart(
                financial_summary
            )

        else:

            st.info(
                "Amount column was not detected."
            )

    st.divider()

    # --------------------------------------------------------
    # AMOUNT TREND
    # --------------------------------------------------------

    if (
        "date" in df.columns
        and "amount" in df.columns
    ):

        st.subheader(
            "📈 Financial Activity Over Time"
        )

        trend_df = (
            df.dropna(
                subset=["date", "amount"]
            )
            .groupby("date")["amount"]
            .sum()
            .sort_index()
        )

        if len(trend_df) > 0:

            st.line_chart(
                trend_df
            )

    st.divider()

    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    st.subheader(
        "⚠️ High-Risk Transactions"
    )

    high_risk = (
        df[
            df["risk_level"].isin(
                ["HIGH", "CRITICAL"]
            )
        ]
        .sort_values(
            "anomaly_score",
            ascending=False
        )
    )

    if len(high_risk) > 0:

        st.dataframe(
            high_risk,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No high-risk transactions detected."
        )


# ============================================================
# TRANSACTION EXPLORER
# ============================================================

elif page == "🔎 Transaction Explorer":

    st.header("🔎 Transaction Explorer")

    if df is None:

        st.info(
            "Upload or load financial data first."
        )

        st.stop()

    st.write(
        "Search and examine individual financial records."
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        risk_filter = st.multiselect(
            "Filter by risk",
            [
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW"
            ],
            default=[
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW"
            ]
        )

    with col2:

        search_text = st.text_input(
            "Search transaction"
        )

    filtered_df = df[
        df["risk_level"].isin(
            risk_filter
        )
    ].copy()

    if search_text:

        mask = (
            filtered_df
            .astype(str)
            .apply(
                lambda row:
                row.str.contains(
                    search_text,
                    case=False,
                    na=False
                ).any(),
                axis=1
            )
        )

        filtered_df = filtered_df[
            mask
        ]

    st.write(
        f"Showing **{len(filtered_df)}** records"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # SINGLE TRANSACTION
    # --------------------------------------------------------

    st.subheader(
        "🔍 Examine a Single Transaction"
    )

    if len(filtered_df) == 0:

        st.warning(
            "No transactions match the current filters."
        )

    else:

        selected_index = st.selectbox(
            "Select a transaction",
            filtered_df.index,
            format_func=lambda x:
            f"Record {x + 1}"
        )

        selected_row = filtered_df.loc[
            selected_index
        ]

        st.markdown("### Transaction Details")

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:

            st.write(
                "**Record ID:**",
                selected_index
            )

            if "transaction_id" in selected_row.index:

                st.write(
                    "**Transaction ID:**",
                    selected_row["transaction_id"]
                )

            if "order_id" in selected_row.index:

                st.write(
                    "**Order ID:**",
                    selected_row["order_id"]
                )

            if "customer_id" in selected_row.index:

                st.write(
                    "**Customer ID:**",
                    selected_row["customer_id"]
                )

        with detail_col2:

            if "amount" in selected_row.index:

                st.write(
                    "**Amount:**",
                    selected_row["amount"]
                )

            st.write(
                "**Risk:**",
                selected_row["risk_level"]
            )

            st.write(
                "**Anomaly Score:**",
                selected_row["anomaly_score"]
            )

        st.divider()

        with st.expander(
            "View complete transaction record"
        ):

            st.dataframe(
                selected_row.to_frame(
                    "Value"
                ),
                use_container_width=True
            )

        # ----------------------------------------------------
        # AI SINGLE TRANSACTION
        # ----------------------------------------------------

        st.subheader(
            "🤖 AI Transaction Investigation"
        )

        if not gemini_available:

            st.warning(
                "Gemini API is not connected. "
                "Add GEMINI_API_KEY to Streamlit Secrets."
            )

        else:

            if st.button(
                "Investigate This Transaction",
                type="primary"
            ):

                with st.spinner(
                    "AI is examining this transaction..."
                ):

                    try:

                        result = (
                            investigate_single_transaction(
                                selected_row
                            )
                        )

                        st.success(
                            "Transaction investigation completed."
                        )

                        st.text_area(
                            "AI Investigation",
                            result,
                            height=450
                        )

                    except Exception as e:

                        st.error(
                            f"AI investigation failed: {e}"
                        )


# ============================================================
# AI INVESTIGATION
# ============================================================

elif page == "🤖 AI Investigation":

    st.header(
        "🤖 AI Financial Investigation"
    )

    if df is None:

        st.info(
            "Upload or load financial data first."
        )

        st.stop()

    st.write(
        "LedgerLens analyzes the available financial evidence "
        "and generates an investigation report for human review."
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    high_risk = df[
        df["risk_level"].isin(
            ["HIGH", "CRITICAL"]
        )
    ]

    st.info(
        f"LedgerLens has identified "
        f"**{len(high_risk)} high/critical-risk records** "
        f"out of **{len(df)} total records**."
    )

    # --------------------------------------------------------
    # AI BUTTON
    # --------------------------------------------------------

    if not gemini_available:

        st.error(
            "Gemini API is not connected."
        )

        st.code(
            'GEMINI_API_KEY = "YOUR_API_KEY"'
        )

        st.stop()

    if st.button(
        "🚀 Run Full AI Investigation",
        type="primary"
    ):

        prompt = make_ai_prompt(df)

        with st.spinner(
            "LedgerLens AI is investigating the financial data..."
        ):

            try:

                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=prompt
                )

                st.success(
                    "AI investigation completed."
                )

                st.subheader(
                    "📋 Investigation Report"
                )

                st.text_area(
                    "LedgerLens AI Report",
                    response.text,
                    height=600
                )

            except Exception as e:

                st.error(
                    f"AI investigation failed: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "LedgerLens • AI Finance Controller • "
    "Recommendations are for human review only."
)
