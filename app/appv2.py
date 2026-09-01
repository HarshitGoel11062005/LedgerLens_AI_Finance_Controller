import streamlit as st
import pandas as pd
import numpy as np
import io
import re

from pypdf import PdfReader
from google import genai


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LedgerLens | AI Finance Controller",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# GEMINI CONNECTION
# =========================================================

try:
    api_key = st.secrets["GEMINI_API_KEY"]

    client = genai.Client(
        api_key=api_key
    )

    gemini_available = True

except Exception:
    client = None
    gemini_available = False


# =========================================================
# SESSION STATE
# =========================================================

if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}

if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = None

if "selected_transaction" not in st.session_state:
    st.session_state.selected_transaction = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("💰 LedgerLens")

    st.caption(
        "AI Finance Controller"
    )

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Go to",
        [
            "📊 Dashboard",
            "📂 Data Upload",
            "🔎 Transaction Explorer",
            "🤖 AI Investigation"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("System Status")

    if gemini_available:
        st.success(
            "🟢 Gemini AI Connected"
        )
    else:
        st.warning(
            "🟡 Gemini AI Not Configured"
        )

    with st.container(border=True):

        st.markdown("### ⚙️ LedgerLens Engine")

        st.caption(
            "Financial data processing"
        )

        st.caption(
            "Anomaly detection"
        )

        st.caption(
            "Risk classification"
        )

        st.caption(
            "Transaction investigation"
        )

        st.caption(
            "AI explanations"
        )

    st.divider()

    st.subheader("Capabilities")

    st.caption("📊 Financial risk dashboard")
    st.caption("🚨 High-risk detection")
    st.caption("🔎 Transaction investigation")
    st.caption("🤖 AI-powered explanations")
    st.caption("📄 PDF financial extraction")
    st.caption("📥 CSV export")

    st.divider()

    st.caption(
        "LedgerLens v2.0"
    )


# =========================================================
# HERO SECTION
# =========================================================

st.title("💰 LedgerLens")

st.subheader(
    "AI-powered financial risk intelligence and investigation platform"
)

st.caption(
    "Automated Risk Detection • Financial Analytics • "
    "Transaction Investigation • AI Explanations"
)

st.divider()


# =========================================================
# DATASET DETECTION
# =========================================================

def detect_dataset_type(df):

    columns = [
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for col in df.columns
    ]

    text = " ".join(columns)

    if any(
        x in text
        for x in [
            "transaction_id",
            "payment_id",
            "order_id"
        ]
    ):
        return "Transactions"

    if any(
        x in text
        for x in [
            "refund_id",
            "refund_amount",
            "refund_date"
        ]
    ):
        return "Refunds"

    if any(
        x in text
        for x in [
            "fee_amount",
            "processing_fee",
            "fee_id"
        ]
    ):
        return "Fees"

    if any(
        x in text
        for x in [
            "settlement_id",
            "settlement_amount",
            "settlement_date"
        ]
    ):
        return "Settlements"

    return "Financial Data"


# =========================================================
# NORMALIZE COLUMNS
# =========================================================

def normalize_columns(df):

    df = df.copy()

    df.columns = [
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for col in df.columns
    ]

    return df


# =========================================================
# FIND AMOUNT COLUMN
# =========================================================

def find_amount_column(df):

    priority = [
        "amount",
        "transaction_amount",
        "payment_amount",
        "refund_amount",
        "fee_amount",
        "settlement_amount",
        "total_amount",
        "amount_inr",
        "value"
    ]

    for column in priority:

        if column in df.columns:
            return column

    for column in df.columns:

        if (
            "amount" in column
            or "value" in column
        ):
            return column

    return None


# =========================================================
# ANOMALY ANALYSIS
# =========================================================

def analyze_financial_data(df):

    df = normalize_columns(df)

    amount_column = find_amount_column(df)

    if amount_column is None:

        return (
            None,
            "No financial amount column could be identified."
        )

    df[amount_column] = pd.to_numeric(
        df[amount_column],
        errors="coerce"
    )

    valid_df = df.dropna(
        subset=[amount_column]
    ).copy()

    if len(valid_df) == 0:

        return (
            None,
            "No valid financial amounts were found."
        )

    mean_amount = valid_df[
        amount_column
    ].mean()

    std_amount = valid_df[
        amount_column
    ].std()

    # -----------------------------------------------------
    # ANOMALY SCORE
    # -----------------------------------------------------

    if (
        std_amount == 0
        or pd.isna(std_amount)
    ):

        valid_df["anomaly_score"] = 0.0

    else:

        valid_df["anomaly_score"] = (
            (
                (
                    valid_df[amount_column]
                    - mean_amount
                ).abs()
                / std_amount
            )
            * 25
        )

        valid_df["anomaly_score"] = (
            valid_df["anomaly_score"]
            .clip(0, 100)
        )

    # -----------------------------------------------------
    # RISK CLASSIFICATION
    # -----------------------------------------------------

    def risk_level(score):

        if score >= 75:
            return "CRITICAL"

        elif score >= 50:
            return "HIGH"

        elif score >= 25:
            return "MEDIUM"

        return "LOW"

    valid_df["risk_level"] = (
        valid_df["anomaly_score"]
        .apply(risk_level)
    )

    return valid_df, None


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):

    uploaded_file.seek(0)

    reader = PdfReader(
        uploaded_file
    )

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    return text


# =========================================================
# PDF TABLE CONVERSION
# =========================================================

def pdf_text_to_dataframe(text):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        return None

    # =====================================================
    # PIPE TABLE
    # =====================================================

    pipe_lines = [
        line
        for line in lines
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

            data_rows = rows[1:]

            if data_rows:

                first_row = data_rows[0]

                if all(
                    re.fullmatch(
                        r"[-: ]+",
                        item or ""
                    )
                    for item in first_row
                ):

                    data_rows = data_rows[1:]

            width = len(header)

            cleaned_rows = [

                row[:width]
                + [""] * max(
                    0,
                    width - len(row)
                )

                for row in data_rows
            ]

            if cleaned_rows:

                return pd.DataFrame(
                    cleaned_rows,
                    columns=header
                )

    # =====================================================
    # COMMA TABLE
    # =====================================================

    comma_lines = [
        line
        for line in lines
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

    # =====================================================
    # WHITESPACE TABLE
    # =====================================================

    rows = []

    for line in lines:

        parts = re.split(
            r"\s{2,}",
            line
        )

        if len(parts) >= 2:

            rows.append(parts)

    if len(rows) >= 3:

        header = rows[0]

        data_rows = rows[1:]

        width = len(header)

        cleaned_rows = [

            row[:width]
            + [""] * max(
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
# DASHBOARD
# =========================================================

def display_dashboard(
    df,
    dataset_name="Financial Data"
):

    analyzed_df, error = (
        analyze_financial_data(df)
    )

    if error:

        st.warning(error)

        st.dataframe(
            df,
            use_container_width=True
        )

        return analyzed_df

    # =====================================================
    # DASHBOARD HEADER
    # =====================================================

    with st.container(border=True):

        st.header(
            "📊 Financial Risk Dashboard"
        )

        st.caption(
            f"Dataset detected as: {dataset_name}"
        )

    # =====================================================
    # METRICS
    # =====================================================

    total_records = len(
        analyzed_df
    )

    critical_count = len(
        analyzed_df[
            analyzed_df["risk_level"]
            == "CRITICAL"
        ]
    )

    high_count = len(
        analyzed_df[
            analyzed_df["risk_level"]
            == "HIGH"
        ]
    )

    medium_count = len(
        analyzed_df[
            analyzed_df["risk_level"]
            == "MEDIUM"
        ]
    )

    low_count = len(
        analyzed_df[
            analyzed_df["risk_level"]
            == "LOW"
        ]
    )

    high_critical_count = (
        critical_count
        + high_count
    )

    amount_column = find_amount_column(
        analyzed_df
    )

    total_amount = (
        analyzed_df[amount_column].sum()
        if amount_column
        else 0
    )

    # =====================================================
    # KPI ROW 1
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "TOTAL RECORDS",
            f"{total_records:,}",
            help="Total financial records analyzed."
        )

    with c2:

        st.metric(
            "🔴 HIGH / CRITICAL",
            f"{high_critical_count:,}",
            help="Records requiring human review."
        )

    with c3:

        st.metric(
            "🟠 MEDIUM RISK",
            f"{medium_count:,}",
            help="Records requiring additional attention."
        )

    with c4:

        st.metric(
            "🟢 LOW RISK",
            f"{low_count:,}",
            help="Records with no major detected anomaly."
        )

    st.write("")

    # =====================================================
    # KPI ROW 2
    # =====================================================

    with st.container(border=True):

        st.subheader(
            "💰 Financial Overview"
        )

        a1, a2, a3, a4 = st.columns(4)

        with a1:

            st.metric(
                "Total Amount",
                f"{total_amount:,.2f}"
            )

        with a2:

            st.metric(
                "Average Amount",
                (
                    f"{analyzed_df[amount_column].mean():,.2f}"
                    if amount_column
                    else "N/A"
                )
            )

        with a3:

            st.metric(
                "Maximum Amount",
                (
                    f"{analyzed_df[amount_column].max():,.2f}"
                    if amount_column
                    else "N/A"
                )
            )

        with a4:

            st.metric(
                "Critical Records",
                f"{critical_count:,}"
            )

    st.write("")

    # =====================================================
    # RISK DISTRIBUTION
    # =====================================================

    left, right = st.columns(2)

    with left:

        with st.container(border=True):

            st.subheader(
                "🎯 Risk Distribution"
            )

            st.caption(
                "Distribution of financial risk levels."
            )

            risk_chart = pd.DataFrame(
                {
                    "Risk Level": [
                        "Critical",
                        "High",
                        "Medium",
                        "Low"
                    ],

                    "Records": [
                        critical_count,
                        high_count,
                        medium_count,
                        low_count
                    ]
                }
            )

            st.bar_chart(
                risk_chart.set_index(
                    "Risk Level"
                ),
                use_container_width=True
            )

    # =====================================================
    # FINANCIAL OVERVIEW CHART
    # =====================================================

    with right:

        with st.container(border=True):

            st.subheader(
                "💰 Amount Overview"
            )

            st.caption(
                "Financial amount across analyzed records."
            )

            if amount_column:

                amount_chart = (
                    analyzed_df[
                        [amount_column]
                    ]
                    .reset_index(drop=True)
                )

                amount_chart.columns = [
                    "Amount"
                ]

                st.line_chart(
                    amount_chart,
                    use_container_width=True
                )

            else:

                st.info(
                    "No amount column available."
                )

    st.write("")

    # =====================================================
    # ANOMALY SCORE DISTRIBUTION
    # =====================================================

    with st.container(border=True):

        st.subheader(
            "📊 Anomaly Score Distribution"
        )

        st.caption(
            "Higher scores indicate stronger deviation "
            "from normal financial activity."
        )

        score_chart = pd.DataFrame(
            {
                "Anomaly Score":
                    analyzed_df[
                        "anomaly_score"
                    ]
            }
        )

        st.area_chart(
            score_chart,
            use_container_width=True
        )

    st.write("")

    # =====================================================
    # FINANCIAL DATA
    # =====================================================

    with st.container(border=True):

        st.subheader(
            "📋 Financial Data Analysis"
        )

        st.caption(
            "Complete processed financial dataset."
        )

        st.dataframe(
            analyzed_df,
            use_container_width=True,
            height=420
        )

    st.write("")

    # =====================================================
    # HIGH-RISK RECORDS
    # =====================================================

    high_risk = (
        analyzed_df[
            analyzed_df["risk_level"]
            .isin(
                ["HIGH", "CRITICAL"]
            )
        ]
        .sort_values(
            "anomaly_score",
            ascending=False
        )
    )

    with st.container(border=True):

        st.subheader(
            "🚨 High-Risk Records"
        )

        st.caption(
            "Transactions requiring the highest level "
            "of human review."
        )

        if len(high_risk) > 0:

            st.warning(
                f"{len(high_risk)} high/critical "
                "record(s) detected."
            )

            st.dataframe(
                high_risk,
                use_container_width=True,
                height=350
            )

        else:

            st.success(
                "No high-risk financial records detected."
            )

    st.write("")

    # =====================================================
    # DOWNLOAD
    # =====================================================

    with st.container(border=True):

        st.subheader(
            "📥 Export Analysis"
        )

        st.caption(
            "Download the processed financial dataset."
        )

        csv_data = analyzed_df.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Processed CSV",
            data=csv_data,
            file_name=(
                "LedgerLens_Processed_Data.csv"
            ),
            mime="text/csv"
        )

    return analyzed_df


# =========================================================
# TRANSACTION EXPLORER
# =========================================================

def transaction_explorer(df):

    analyzed_df, error = (
        analyze_financial_data(df)
    )

    if error:

        st.error(error)

        return

    st.header(
        "🔎 Transaction Explorer"
    )

    st.caption(
        "Search and examine individual financial "
        "records separately."
    )

    # =====================================================
    # SEARCH
    # =====================================================

    search_text = st.text_input(
        "🔍 Search transaction",
        placeholder=(
            "Search transaction ID, customer ID, "
            "order ID..."
        )
    )

    filtered_df = analyzed_df.copy()

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

        filtered_df = (
            filtered_df[mask]
        )

    if len(filtered_df) == 0:

        st.warning(
            "No matching transaction found."
        )

        return

    # =====================================================
    # CREATE RECORD OPTIONS
    # =====================================================

    display_options = []

    for index, row in filtered_df.iterrows():

        transaction_id = None

        for column in [
            "transaction_id",
            "payment_id",
            "order_id",
            "refund_id",
            "settlement_id"
        ]:

            if column in row.index:

                transaction_id = (
                    str(row[column])
                )

                break

        if transaction_id is None:

            transaction_id = (
                f"Record {index}"
            )

        display_options.append(
            (index, transaction_id)
        )

    # =====================================================
    # SELECT RECORD
    # =====================================================

    selected_label = st.selectbox(
        "Select a financial record",
        [
            label
            for _, label
            in display_options
        ]
    )

    selected_index = next(
        index
        for index, label
        in display_options
        if label == selected_label
    )

    selected = analyzed_df.loc[
        selected_index
    ]

    st.divider()

    # =====================================================
    # RISK STATUS
    # =====================================================

    risk = selected[
        "risk_level"
    ]

    score = selected[
        "anomaly_score"
    ]

    if risk == "CRITICAL":

        st.error(
            f"🔴 CRITICAL RISK • "
            f"Anomaly Score: {score:.2f}"
        )

    elif risk == "HIGH":

        st.warning(
            f"🟠 HIGH RISK • "
            f"Anomaly Score: {score:.2f}"
        )

    elif risk == "MEDIUM":

        st.warning(
            f"🟡 MEDIUM RISK • "
            f"Anomaly Score: {score:.2f}"
        )

    else:

        st.success(
            f"🟢 LOW RISK • "
            f"Anomaly Score: {score:.2f}"
        )

    # =====================================================
    # RECORD SUMMARY
    # =====================================================

    with st.container(border=True):

        st.subheader(
            "📄 Record Details"
        )

        details = pd.DataFrame(
            {
                "Field":
                    selected.index,

                "Value":
                    [
                        str(value)
                        for value
                        in selected.values
                    ]
            }
        )

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # AI INVESTIGATION
    # =====================================================

    st.subheader(
        "🤖 AI Investigation"
    )

    if not gemini_available:

        st.info(
            "Gemini AI is not configured. "
            "Add GEMINI_API_KEY to Streamlit Secrets "
            "to enable AI investigation."
        )

        return

    investigate = st.button(
        "🔎 Investigate This Record",
        type="primary"
    )

    if investigate:

        evidence = "\n".join(
            [
                f"{column}: {value}"
                for column, value
                in selected.items()
            ]
        )

        prompt = f"""
You are LedgerLens, an AI financial investigation assistant.

Analyze ONLY the financial record below.

Rules:

1. Identify suspicious or unusual activity.
2. Explain the evidence clearly.
3. Mention financial impact only when supported.
4. Do not invent information.
5. Clearly state when evidence is insufficient.
6. Do not authorize or execute transactions.
7. Recommendations are for human review only.
8. Be concise and professional.

Return exactly:

Finding:
Evidence:
Financial Impact:
Possible Cause:
Recommendation:
Confidence:

FINANCIAL RECORD:

{evidence}
"""

        with st.spinner(
            "LedgerLens AI is investigating this record..."
        ):

            try:

                response = (
                    client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=prompt
                    )
                )

                st.success(
                    "Investigation completed."
                )

                with st.container(border=True):

                    st.subheader(
                        "📋 AI Investigation Report"
                    )

                    st.text_area(
                        "LedgerLens AI Report",
                        response.text,
                        height=450
                    )

            except Exception as e:

                st.error(
                    f"AI investigation failed: {e}"
                )


# =========================================================
# DATA UPLOAD PAGE
# =========================================================

if page == "📂 Data Upload":

    st.header(
        "📂 Upload Financial Data"
    )

    st.write(
        "Upload one or more financial CSV or PDF files. "
        "LedgerLens automatically identifies the dataset "
        "and generates the appropriate analysis."
    )

    st.info(
        "💡 You do NOT need to tell LedgerLens whether "
        "the file is transactions, refunds, fees, or settlements."
    )

    uploaded_files = st.file_uploader(
        "Choose financial files",
        type=["csv", "pdf"],
        accept_multiple_files=True
    )

    if not uploaded_files:

        st.info(
            "Upload one or more financial files "
            "to begin analysis."
        )

    else:

        st.success(
            f"{len(uploaded_files)} file(s) uploaded."
        )

        for uploaded_file in uploaded_files:

            st.divider()

            file_name = (
                uploaded_file.name.lower()
            )

            # =================================================
            # CSV
            # =================================================

            if file_name.endswith(".csv"):

                try:

                    df = pd.read_csv(
                        uploaded_file
                    )

                    df = normalize_columns(
                        df
                    )

                    dataset_type = (
                        detect_dataset_type(
                            df
                        )
                    )

                    analyzed_df, error = (
                        analyze_financial_data(
                            df
                        )
                    )

                    if error:

                        st.error(error)

                    else:

                        st.success(
                            f"Detected dataset: "
                            f"{dataset_type}"
                        )

                        st.session_state.processed_files[
                            uploaded_file.name
                        ] = {

                            "data":
                                analyzed_df,

                            "type":
                                dataset_type
                        }

                        with st.expander(
                            f"📄 {uploaded_file.name}",
                            expanded=True
                        ):

                            st.caption(
                                f"Rows: {len(df)} | "
                                f"Columns: {len(df.columns)}"
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

                    extracted_text = (
                        extract_pdf_text(
                            uploaded_file
                        )
                    )

                    if not extracted_text.strip():

                        st.warning(
                            "No readable text was found "
                            "in this PDF."
                        )

                        continue

                    st.success(
                        "PDF text extracted successfully."
                    )

                    pdf_df = (
                        pdf_text_to_dataframe(
                            extracted_text
                        )
                    )

                    with st.expander(
                        f"📄 {uploaded_file.name}",
                        expanded=True
                    ):

                        # =====================================
                        # TABLE FOUND
                        # =====================================

                        if pdf_df is not None:

                            pdf_df = normalize_columns(
                                pdf_df
                            )

                            dataset_type = (
                                detect_dataset_type(
                                    pdf_df
                                )
                            )

                            st.success(
                                "Structured financial table detected."
                            )

                            st.subheader(
                                "📊 Extracted Financial Table"
                            )

                            st.dataframe(
                                pdf_df,
                                use_container_width=True
                            )

                            st.subheader(
                                "📊 Generated Dashboard"
                            )

                            analyzed_pdf_df = (
                                display_dashboard(
                                    pdf_df,
                                    dataset_type
                                )
                            )

                            if analyzed_pdf_df is not None:

                                st.session_state.processed_files[
                                    uploaded_file.name
                                ] = {

                                    "data":
                                        analyzed_pdf_df,

                                    "type":
                                        dataset_type
                                }

                            # =================================
                            # CONVERT PDF TO CSV
                            # =================================

                            converted_csv = (
                                pdf_df.to_csv(
                                    index=False
                                )
                            )

                            st.download_button(
                                "⬇️ Download Converted CSV",
                                data=converted_csv,
                                file_name=(
                                    uploaded_file.name
                                    .rsplit(".", 1)[0]
                                    + "_converted.csv"
                                ),
                                mime="text/csv"
                            )

                        # =====================================
                        # NO TABLE
                        # =====================================

                        else:

                            st.info(
                                "No structured financial table "
                                "was detected."
                            )

                            st.subheader(
                                "📑 Extracted PDF Content"
                            )

                            st.text_area(
                                "PDF Content",
                                extracted_text,
                                height=350
                            )

                            st.info(
                                "AI investigation can still "
                                "analyze this financial report."
                            )

                            if gemini_available:

                                investigate_pdf = st.button(
                                    "🔎 Investigate PDF",
                                    key=(
                                        "pdf_ai_"
                                        + uploaded_file.name
                                    )
                                )

                                if investigate_pdf:

                                    prompt = f"""
You are LedgerLens, an AI financial investigation assistant.

Analyze ONLY the financial evidence below.

Identify:

- Suspicious activity
- Unusual financial patterns
- Possible financial impact
- Possible causes
- Recommendations
- Confidence

Do not invent missing information.

FINANCIAL EVIDENCE:

{extracted_text}
"""

                                    with st.spinner(
                                        "LedgerLens AI is investigating..."
                                    ):

                                        try:

                                            response = (
                                                client.models.generate_content(
                                                    model="gemini-3.5-flash",
                                                    contents=prompt
                                                )
                                            )

                                            st.success(
                                                "Investigation completed."
                                            )

                                            with st.container(
                                                border=True
                                            ):

                                                st.subheader(
                                                    "📋 AI Investigation Report"
                                                )

                                                st.text_area(
                                                    "AI Investigation Report",
                                                    response.text,
                                                    height=450
                                                )

                                        except Exception as e:

                                            st.error(
                                                f"AI investigation failed: {e}"
                                            )

                            else:

                                st.warning(
                                    "Gemini AI is not configured."
                                )

                except Exception as e:

                    st.error(
                        f"Error processing PDF: {e}"
                    )


# =========================================================
# DASHBOARD PAGE
# =========================================================

elif page == "📊 Dashboard":

    st.header(
        "📊 LedgerLens Dashboard"
    )

    if not st.session_state.processed_files:

        st.info(
            "No financial dataset is currently loaded."
        )

        with st.container(border=True):

            st.subheader(
                "🚀 How to begin"
            )

            st.markdown(
                """
                **1. Upload** your CSV or PDF

                **2. Detect** the financial dataset

                **3. Analyze** financial amounts

                **4. Detect** anomalies

                **5. Classify** LOW, MEDIUM, HIGH or CRITICAL risk

                **6. Investigate** individual transactions

                **7. Use AI** to explain suspicious activity

                **8. Export** the processed dataset
                """
            )

    else:

        dataset_names = list(
            st.session_state.processed_files.keys()
        )

        selected_file = st.selectbox(
            "Select dataset",
            dataset_names
        )

        selected_data = (
            st.session_state.processed_files[
                selected_file
            ]
        )

        display_dashboard(
            selected_data["data"],
            selected_data["type"]
        )


# =========================================================
# TRANSACTION EXPLORER PAGE
# =========================================================

elif page == "🔎 Transaction Explorer":

    if not st.session_state.processed_files:

        st.info(
            "Upload a financial dataset first."
        )

    else:

        dataset_names = list(
            st.session_state.processed_files.keys()
        )

        selected_file = st.selectbox(
            "Select dataset",
            dataset_names,
            key="explorer_dataset"
        )

        selected_data = (
            st.session_state.processed_files[
                selected_file
            ]
        )

        transaction_explorer(
            selected_data["data"]
        )


# =========================================================
# AI INVESTIGATION PAGE
# =========================================================

elif page == "🤖 AI Investigation":

    st.header(
        "🤖 AI Financial Investigation"
    )

    if gemini_available:

        st.success(
            "🟢 Gemini AI is connected and ready."
        )

        st.write(
            "Select a dataset and investigate an "
            "individual financial record using LedgerLens AI."
        )

        if st.session_state.processed_files:

            dataset_names = list(
                st.session_state.processed_files.keys()
            )

            selected_file = st.selectbox(
                "Select dataset",
                dataset_names,
                key="ai_dataset"
            )

            selected_data = (
                st.session_state.processed_files[
                    selected_file
                ]
            )

            transaction_explorer(
                selected_data["data"]
            )

        else:

            st.info(
                "Upload financial data first."
            )

    else:

        st.error(
            "Gemini AI is not configured."
        )

        st.write(
            "Add GEMINI_API_KEY to Streamlit Secrets."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "LedgerLens • AI Finance Controller • "
    "Financial Risk Intelligence Platform"
)

st.caption(
    "Automated analysis • Risk detection • "
    "Transaction investigation • AI explanations"
)
