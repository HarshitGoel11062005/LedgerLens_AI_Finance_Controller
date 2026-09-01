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
    page_title="LedgerLens | AI Finance Controller",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM UI / CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #f5f7fb;
    }

    .main {
        padding-top: 1rem;
    }

    h1, h2, h3 {
        color: #172033;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }

    section[data-testid="stSidebar"] * {
        color: #f9fafb !important;
    }

    .sidebar-logo {
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        font-size: 13px;
        color: #9ca3af !important;
        margin-bottom: 30px;
    }

    .status-box {
        background: #1f2937;
        padding: 15px;
        border-radius: 12px;
        margin-top: 20px;
        border: 1px solid #374151;
    }

    /* ---------- HERO ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #111827,
            #1e3a5f
        );

        padding: 32px 35px;
        border-radius: 18px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
    }

    .hero-title {
        color: white !important;
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        color: #dbeafe !important;
        font-size: 17px;
        margin-bottom: 0;
    }

    /* ---------- KPI CARDS ---------- */

    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
        min-height: 120px;
    }

    .kpi-label {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }

    .kpi-value {
        color: #111827;
        font-size: 31px;
        font-weight: 800;
        margin-top: 8px;
    }

    .kpi-description {
        color: #94a3b8;
        font-size: 12px;
        margin-top: 4px;
    }

    /* ---------- SECTION ---------- */

    .section-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: #172033;
        margin-bottom: 5px;
    }

    .section-description {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 18px;
    }

    /* ---------- RISK BADGES ---------- */

    .risk-critical {
        color: #991b1b;
        font-weight: 800;
    }

    .risk-high {
        color: #c2410c;
        font-weight: 800;
    }

    .risk-medium {
        color: #a16207;
        font-weight: 800;
    }

    .risk-low {
        color: #15803d;
        font-weight: 800;
    }

    /* ---------- UPLOAD AREA ---------- */

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 16px;
        border: 2px dashed #cbd5e1;
        padding: 10px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
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
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-logo">💰 LedgerLens</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'AI Finance Controller'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Navigation")

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "📂 Data Upload",
            "🤖 AI Investigation"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### System Status")

    if gemini_available:

        st.success("Gemini AI Connected")

    else:

        st.warning("Gemini AI Not Configured")

    st.markdown(
        """
        <div class="status-box">
            <b>LedgerLens Engine</b><br>
            <span style="color:#9ca3af;">
            Financial anomaly detection<br>
            Risk classification<br>
            AI investigation
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            LedgerLens
        </div>

        <div class="hero-subtitle">
            AI-powered financial risk intelligence and investigation platform
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DATASET TYPE DETECTION
# =========================================================

def detect_dataset_type(df):

    columns = [
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        for col in df.columns
    ]

    column_text = " ".join(columns)

    if any(
        keyword in column_text
        for keyword in [
            "transaction_id",
            "payment_id",
            "order_id"
        ]
    ):
        return "Transactions"

    if any(
        keyword in column_text
        for keyword in [
            "refund_id",
            "refund_amount",
            "refund_date"
        ]
    ):
        return "Refunds"

    if any(
        keyword in column_text
        for keyword in [
            "fee",
            "fee_amount",
            "processing_fee"
        ]
    ):
        return "Fees"

    if any(
        keyword in column_text
        for keyword in [
            "settlement_id",
            "settlement_amount",
            "settlement_date"
        ]
    ):
        return "Settlements"

    if "amount" in columns:
        return "Financial Data"

    return "Financial Data"


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


    # -----------------------------------------------------
    # PIPE-SEPARATED TABLE
    # -----------------------------------------------------

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

            if len(rows) > 1:

                if all(
                    re.fullmatch(
                        r"[-: ]+",
                        item or ""
                    )
                    for item in rows[1]
                ):

                    rows = rows[2:]

                else:

                    rows = rows[1:]

            if rows:

                width = len(header)

                cleaned_rows = [

                    row[:width]
                    + [""] * max(
                        0,
                        width - len(row)
                    )

                    for row in rows

                ]

                return pd.DataFrame(
                    cleaned_rows,
                    columns=header
                )


    # -----------------------------------------------------
    # COMMA-SEPARATED TABLE
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # WHITESPACE TABLE
    # -----------------------------------------------------

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
# NORMALIZE COLUMNS
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

    amount_column = None

    possible_amount_columns = [

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

    for column in possible_amount_columns:

        if column in df.columns:

            amount_column = column

            break


    if amount_column is None:

        for column in df.columns:

            if (
                "amount" in column
                or "value" in column
            ):

                amount_column = column

                break


    if amount_column is None:

        return None, (
            "No financial amount column "
            "could be identified."
        )


    df[amount_column] = pd.to_numeric(
        df[amount_column],
        errors="coerce"
    )


    valid_df = df.dropna(
        subset=[amount_column]
    ).copy()


    if len(valid_df) == 0:

        return None, (
            "No valid financial amounts "
            "were found."
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
# DASHBOARD
# =========================================================

def display_dashboard(df, dataset_name):

    analyzed_df, error_message = (
        analyze_financial_data(df)
    )


    if error_message:

        st.warning(error_message)

        st.dataframe(
            df,
            use_container_width=True
        )

        return


    # =====================================================
    # DASHBOARD HEADER
    # =====================================================

    st.markdown(
        """
        <div class="section-card">

        <div class="section-title">
        📊 Financial Risk Dashboard
        </div>

        <div class="section-description">
        Automated analysis of the uploaded financial dataset.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # METRICS
    # =====================================================

    total_records = len(analyzed_df)

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


    amount_columns = [

        column
        for column in analyzed_df.columns

        if (
            "amount" in column
            or column == "value"
        )

    ]


    total_amount = 0

    if amount_columns:

        total_amount = analyzed_df[
            amount_columns[0]
        ].sum()


    # =====================================================
    # KPI CARDS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                TOTAL RECORDS
                </div>

                <div class="kpi-value">
                {total_records:,}
                </div>

                <div class="kpi-description">
                Financial records analyzed
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                🔴 HIGH / CRITICAL
                </div>

                <div class="kpi-value">
                {high_risk_count:,}
                </div>

                <div class="kpi-description">
                Requires human review
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                🟠 MEDIUM RISK
                </div>

                <div class="kpi-value">
                {medium_risk_count:,}
                </div>

                <div class="kpi-description">
                Transactions requiring attention
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                🟢 LOW RISK
                </div>

                <div class="kpi-value">
                {low_risk_count:,}
                </div>

                <div class="kpi-description">
                No major anomaly detected
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # RISK OVERVIEW
    # =====================================================

    col_left, col_right = st.columns(
        [1, 1]
    )


    with col_left:

        st.markdown(
            """
            <div class="section-card">

            <div class="section-title">
            🎯 Risk Distribution
            </div>

            <div class="section-description">
            Distribution of detected financial risk levels.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        risk_data = pd.DataFrame(
            {
                "Risk Level": [
                    "Critical / High",
                    "Medium",
                    "Low"
                ],

                "Records": [
                    high_risk_count,
                    medium_risk_count,
                    low_risk_count
                ]
            }
        )

        risk_data = risk_data.set_index(
            "Risk Level"
        )

        st.bar_chart(
            risk_data,
            use_container_width=True
        )


    with col_right:

        st.markdown(
            """
            <div class="section-card">

            <div class="section-title">
            💰 Financial Overview
            </div>

            <div class="section-description">
            Key financial statistics from the uploaded dataset.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        if amount_columns:

            amount_column = amount_columns[0]

            stat1, stat2 = st.columns(2)

            with stat1:

                st.metric(
                    "Total Amount",
                    f"{total_amount:,.2f}"
                )

            with stat2:

                st.metric(
                    "Average Amount",
                    f"{analyzed_df[amount_column].mean():,.2f}"
                )

            st.metric(
                "Maximum Transaction",
                f"{analyzed_df[amount_column].max():,.2f}"
            )

        else:

            st.info(
                "No amount field available "
                "for financial statistics."
            )


    # =====================================================
    # FULL DATA
    # =====================================================

    st.markdown(
        """
        <div class="section-card">

        <div class="section-title">
        📋 Financial Data Analysis
        </div>

        <div class="section-description">
        Complete dataset with anomaly scores and risk classification.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.dataframe(
        analyzed_df,
        use_container_width=True,
        height=420
    )


    # =====================================================
    # HIGH RISK
    # =====================================================

    high_risk = analyzed_df[
        analyzed_df["risk_level"].isin(
            ["HIGH", "CRITICAL"]
        )
    ].sort_values(
        "anomaly_score",
        ascending=False
    )


    st.markdown(
        """
        <div class="section-card">

        <div class="section-title">
        🚨 High-Risk Financial Records
        </div>

        <div class="section-description">
        Records with the strongest anomaly signals.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    if len(high_risk) > 0:

        st.warning(
            f"{len(high_risk)} high/critical "
            "risk record(s) detected."
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


    # =====================================================
    # EXPORT
    # =====================================================

    st.markdown(
        """
        <div class="section-card">

        <div class="section-title">
        📥 Export Analysis
        </div>

        <div class="section-description">
        Download the processed dataset for further analysis.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    csv_data = analyzed_df.to_csv(
        index=False
    )


    st.download_button(
        label="⬇️ Download Processed CSV",
        data=csv_data,
        file_name="LedgerLens_Processed_Data.csv",
        mime="text/csv",
        use_container_width=False
    )


# =========================================================
# DATA UPLOAD SECTION
# =========================================================

if page == "📂 Data Upload":

    st.header("📂 Upload Financial Data")

    st.write(
        "Upload one or more CSV or PDF financial files. "
        "LedgerLens automatically identifies the dataset type "
        "and generates the appropriate financial analysis."
    )

    uploaded_files = st.file_uploader(
        "Choose financial files",
        type=["csv", "pdf"],
        accept_multiple_files=True
    )


    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} file(s) uploaded successfully."
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

                    dataset_type = (
                        detect_dataset_type(df)
                    )

                    st.success(
                        f"Detected dataset: {dataset_type}"
                    )

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


                    with st.expander(
                        f"📄 {uploaded_file.name}",
                        expanded=True
                    ):

                        pdf_df = (
                            pdf_text_to_dataframe(
                                extracted_text
                            )
                        )


                        # =====================================
                        # STRUCTURED TABLE FOUND
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


                            # Generate dashboard

                            display_dashboard(
                                pdf_df,
                                dataset_type
                            )


                            # Convert to CSV

                            converted_csv = (
                                pdf_df.to_csv(
                                    index=False
                                )
                            )


                            st.download_button(
                                label=(
                                    "⬇️ Download Converted CSV"
                                ),

                                data=converted_csv,

                                file_name=(
                                    uploaded_file.name
                                    .replace(
                                        ".pdf",
                                        ""
                                    )
                                    + "_converted.csv"
                                ),

                                mime="text/csv"
                            )


                        # =====================================
                        # NO STRUCTURED TABLE
                        # =====================================

                        else:

                            st.info(
                                "No structured financial table "
                                "was detected in this PDF."
                            )


                            st.subheader(
                                "📑 Extracted Financial Report"
                            )


                            st.text_area(
                                "PDF Content",
                                extracted_text,
                                height=350
                            )


                            # =================================
                            # AI INVESTIGATION
                            # =================================

                            st.subheader(
                                "🤖 AI Financial Investigation"
                            )


                            if not gemini_available:

                                st.error(
                                    "Gemini API is not configured. "
                                    "Add GEMINI_API_KEY to "
                                    "Streamlit Secrets."
                                )

                            else:

                                investigate = st.button(
                                    "🔎 Investigate Financial Report",
                                    key=(
                                        "investigate_"
                                        + uploaded_file.name
                                    )
                                )


                                if investigate:

                                    investigation_prompt = f"""

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
                                        "AI is investigating..."
                                    ):

                                        try:

                                            response = (
                                                client.models.generate_content(
                                                    model="gemini-3.5-flash",
                                                    contents=(
                                                        investigation_prompt
                                                    )
                                                )
                                            )


                                            st.success(
                                                "Investigation completed!"
                                            )


                                            st.subheader(
                                                "📋 AI Investigation Report"
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


    else:

        st.info(
            "Upload your financial CSV/PDF files above "
            "to generate the LedgerLens dashboard."
        )


# =========================================================
# DASHBOARD PAGE
# =========================================================

elif page == "📊 Dashboard":

    st.header("📊 LedgerLens Dashboard")

    st.info(
        "Upload a CSV or PDF from the "
        "'Data Upload' section to generate "
        "your financial risk dashboard."
    )

    st.markdown(
        """
        ### How LedgerLens works

        **1. Upload →** Your financial CSV/PDF

        **2. Detect →** LedgerLens identifies the financial dataset

        **3. Analyze →** Amounts and anomaly signals are calculated

        **4. Classify →** Records receive LOW, MEDIUM, HIGH or CRITICAL risk

        **5. Investigate →** Gemini can explain suspicious financial evidence

        **6. Export →** Download the processed financial dataset
        """
    )


# =========================================================
# AI INVESTIGATION PAGE
# =========================================================

elif page == "🤖 AI Investigation":

    st.header("🤖 AI Financial Investigation")

    if gemini_available:

        st.success(
            "Gemini AI is connected and ready."
        )

        st.write(
            "Upload a financial PDF from the "
            "'Data Upload' section to generate "
            "an AI-powered investigation."
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

st.markdown(
    """
    <div class="footer">
        LedgerLens • AI Finance Controller •
        Financial Risk Intelligence Platform
    </div>
    """,
    unsafe_allow_html=True
)
