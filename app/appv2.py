import streamlit as st
import pandas as pd
import numpy as np
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
        color: #f8fafc;
    }

    /* Sidebar title */
    .sidebar-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .sidebar-subtitle {
        font-size: 13px;
        color: #94a3b8 !important;
        margin-bottom: 25px;
    }

    .sidebar-section {
        font-size: 12px;
        font-weight: 700;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 22px;
        margin-bottom: 8px;
    }

    .ai-connected {
        background-color: #0f302f;
        border: 1px solid #155e59;
        padding: 12px;
        border-radius: 10px;
        margin-top: 12px;
        margin-bottom: 15px;
    }

    .ai-dot {
        color: #22c55e;
        font-size: 14px;
    }

    /* Main header */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #17233c;
        margin-bottom: 0;
    }

    .main-subtitle {
        color: #64748b;
        font-size: 17px;
        margin-top: 4px;
    }

    /* KPI */
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    /* Section cards */
    .section-card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }

    .risk-critical {
        color: #dc2626;
        font-weight: 700;
    }

    .risk-high {
        color: #ea580c;
        font-weight: 700;
    }

    .risk-medium {
        color: #ca8a04;
        font-weight: 700;
    }

    .risk-low {
        color: #16a34a;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">💰 LedgerLens</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'AI Finance Controller'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # 1. LEDGERLENS
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">01 · Platform</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🏦 LedgerLens")

    st.caption(
        "AI-powered financial risk intelligence "
        "and investigation platform."
    )


    # --------------------------------------------------------
    # 2. DATA UPLOAD
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">02 · Input</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 📂 Data Upload")

    st.caption(
        "Upload CSV or PDF financial data."
    )


    # --------------------------------------------------------
    # 3. AI INVESTIGATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">03 · Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🤖 AI Investigation")

    st.caption(
        "Investigate suspicious financial activity."
    )


    # --------------------------------------------------------
    # 4. AI ENGINE
    # --------------------------------------------------------

    st.markdown(
        '<div class="ai-connected">',
        unsafe_allow_html=True
    )

    if AI_CONNECTED:

        st.markdown(
            '<span class="ai-dot">●</span> '
            '<b>AI Engine Connected</b>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            "⚠️ AI Engine Not Connected"
        )

    st.markdown("</div>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # 5. FINANCIAL RISK
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">04 · Risk Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 📊 Financial Risk")

    st.caption(
        "Overall financial risk overview."
    )


    # --------------------------------------------------------
    # 6. RISK TRANSACTIONS
    # --------------------------------------------------------

    st.markdown("### 🚨 Risk Transactions")

    st.caption(
        "Transactions requiring attention."
    )


    # --------------------------------------------------------
    # 7. ANOMALY DETECTION
    # --------------------------------------------------------

    st.markdown("### 🔍 Anomaly Detection")

    st.caption(
        "Identify unusual transaction behavior."
    )


    # --------------------------------------------------------
    # 8. SINGLE TRANSACTION
    # --------------------------------------------------------

    st.markdown("### 🔎 Single Transaction")

    st.caption(
        "Investigate one transaction with AI."
    )


    # --------------------------------------------------------
    # 9. HIGH RISK
    # --------------------------------------------------------

    st.markdown("### 🔴 High-Risk Transactions")

    st.caption(
        "Review high and critical risk activity."
    )


    # --------------------------------------------------------
    # 10. ABOUT
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">05 · Information</div>',
        unsafe_allow_html=True
    )

    st.markdown("### ℹ️ About LedgerLens")

    st.caption(
        "Financial anomaly detection, "
        "risk scoring and AI investigation."
    )

    st.markdown("---")

    st.caption("LedgerLens v2.0")
    st.caption("Human review required for recommendations.")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Financial Risk Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'AI-powered financial anomaly detection and investigation'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("📂 Upload Financial Data")

uploaded_file = st.file_uploader(
    "Upload your financial dataset",
    type=["csv", "pdf"],
    help="LedgerLens automatically analyzes the uploaded financial data."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_risk(df):

    df = df.copy()

    # Find amount column automatically
    amount_column = None

    possible_amount_columns = [
        "amount",
        "amount_inr",
        "transaction_amount",
        "value",
        "total_amount"
    ]

    for column in possible_amount_columns:

        if column in df.columns:
            amount_column = column
            break

    if amount_column is None:
        return df, None

    df[amount_column] = pd.to_numeric(
        df[amount_column],
        errors="coerce"
    )

    mean_amount = df[amount_column].mean()
    std_amount = df[amount_column].std()

    if std_amount == 0 or pd.isna(std_amount):

        df["anomaly_score"] = 0

    else:

        df["anomaly_score"] = (
            (
                (df[amount_column] - mean_amount).abs()
                / std_amount
            ) * 25
        )

        df["anomaly_score"] = df[
            "anomaly_score"
        ].clip(0, 100)

    def risk_level(score):

        if score >= 75:
            return "CRITICAL"

        elif score >= 50:
            return "HIGH"

        elif score >= 25:
            return "MEDIUM"

        return "LOW"

    df["risk_level"] = df[
        "anomaly_score"
    ].apply(risk_level)

    return df, amount_column


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# PROCESS FILE
# ============================================================

if uploaded_file is not None:

    file_name = uploaded_file.name.lower()


    # ========================================================
    # CSV
    # ========================================================

    if file_name.endswith(".csv"):

        try:

            df = pd.read_csv(uploaded_file)

            st.success(
                f"✅ {uploaded_file.name} uploaded successfully"
            )

            df, amount_column = calculate_risk(df)

            if amount_column is None:

                st.warning(
                    "No amount column was detected. "
                    "Dashboard risk scoring requires a financial amount column."
                )

            # ------------------------------------------------
            # DATASET INFORMATION
            # ------------------------------------------------

            st.caption(
                f"Dataset: {uploaded_file.name}"
            )

            st.divider()

            # ------------------------------------------------
            # KPI DASHBOARD
            # ------------------------------------------------

            st.header("📊 Financial Risk Overview")

            total_records = len(df)

            high_count = len(
                df[
                    df["risk_level"].isin(
                        ["HIGH", "CRITICAL"]
                    )
                ]
            )

            medium_count = len(
                df[
                    df["risk_level"] == "MEDIUM"
                ]
            )

            low_count = len(
                df[
                    df["risk_level"] == "LOW"
                ]
            )

            critical_count = len(
                df[
                    df["risk_level"] == "CRITICAL"
                ]
            )


            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Records",
                total_records
            )

            col2.metric(
                "🔴 High / Critical",
                high_count
            )

            col3.metric(
                "🟠 Medium Risk",
                medium_count
            )

            col4.metric(
                "🟢 Low Risk",
                low_count
            )


            # ------------------------------------------------
            # RISK DISTRIBUTION
            # ------------------------------------------------

            st.divider()

            st.header("🎯 Risk Distribution")

            risk_counts = pd.Series({
                "Critical": len(
                    df[df["risk_level"] == "CRITICAL"]
                ),
                "High": len(
                    df[df["risk_level"] == "HIGH"]
                ),
                "Medium": medium_count,
                "Low": low_count
            })

            st.bar_chart(risk_counts)


            # ------------------------------------------------
            # FINANCIAL OVERVIEW
            # ------------------------------------------------

            st.header("💰 Financial Overview")

            if amount_column:

                total_amount = df[
                    amount_column
                ].sum()

                average_amount = df[
                    amount_column
                ].mean()

                maximum_amount = df[
                    amount_column
                ].max()

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Total Financial Value",
                    f"₹{total_amount:,.2f}"
                )

                c2.metric(
                    "Average Transaction",
                    f"₹{average_amount:,.2f}"
                )

                c3.metric(
                    "Largest Transaction",
                    f"₹{maximum_amount:,.2f}"
                )


            # ------------------------------------------------
            # COMPLETE DATA
            # ------------------------------------------------

            st.divider()

            st.header("📋 Financial Data Analysis")

            st.dataframe(
                df,
                use_container_width=True,
                height=420
            )


            # =================================================
            # ANOMALY DETECTION
            # =================================================

            st.divider()

            st.header("🔍 Anomaly Detection")

            st.write(
                "LedgerLens identifies transactions whose "
                "financial values significantly differ from "
                "the normal transaction pattern."
            )

            anomaly_df = df.sort_values(
                "anomaly_score",
                ascending=False
            )

            st.dataframe(
                anomaly_df,
                use_container_width=True,
                height=350
            )


            # =================================================
            # HIGH RISK TRANSACTIONS
            # =================================================

            st.divider()

            st.header("🚨 High-Risk Transactions")

            high_risk_df = df[
                df["risk_level"].isin(
                    ["HIGH", "CRITICAL"]
                )
            ].sort_values(
                "anomaly_score",
                ascending=False
            )

            if len(high_risk_df) > 0:

                st.warning(
                    f"{len(high_risk_df)} transaction(s) "
                    "require human review."
                )

                st.dataframe(
                    high_risk_df,
                    use_container_width=True,
                    height=350
                )

            else:

                st.success(
                    "No high-risk transactions detected."
                )


            # =================================================
            # SINGLE TRANSACTION INVESTIGATION
            # =================================================

            st.divider()

            st.header(
                "🔎 Single Transaction Investigation"
            )

            st.write(
                "Select one transaction and ask LedgerLens "
                "AI to investigate it separately."
            )

            transaction_options = list(
                range(len(df))
            )

            selected_index = st.selectbox(
                "Select transaction",
                transaction_options
            )

            selected_transaction = df.iloc[
                selected_index
            ]

            st.dataframe(
                selected_transaction.to_frame(
                    "Value"
                ),
                use_container_width=True
            )


            if st.button(
                "🔎 Investigate Selected Transaction",
                use_container_width=True
            ):

                if not AI_CONNECTED:

                    st.error(
                        "Gemini AI is not connected."
                    )

                else:

                    transaction_text = (
                        selected_transaction
                        .to_string()
                    )

                    single_prompt = f"""
You are LedgerLens, an AI financial
investigation assistant.

Investigate ONLY the transaction below.

Transaction:
{transaction_text}

Provide:

1. Risk Assessment
2. Suspicious Indicators
3. Financial Evidence
4. Possible Cause
5. Recommended Human Review
6. Confidence

Do not invent information.
Do not authorize transactions.
Recommendations are for human review only.
"""

                    with st.spinner(
                        "🤖 AI investigating transaction..."
                    ):

                        try:

                            response = client.models.generate_content(
                                model="gemini-3.5-flash",
                                contents=single_prompt
                            )

                            st.success(
                                "Transaction investigation completed."
                            )

                            st.text_area(
                                "AI Transaction Investigation",
                                response.text,
                                height=400
                            )

                        except Exception as e:

                            st.error(
                                f"AI investigation failed: {e}"
                            )


            # =================================================
            # FULL AI INVESTIGATION
            # =================================================

            st.divider()

            st.header(
                "🤖 AI Financial Investigation"
            )

            st.write(
                "Run a complete AI investigation of the "
                "uploaded financial dataset."
            )

            if st.button(
                "🚀 Run Full AI Investigation",
                use_container_width=True
            ):

                if not AI_CONNECTED:

                    st.error(
                        "Gemini AI is not connected. "
                        "Check GEMINI_API_KEY in Streamlit Secrets."
                    )

                else:

                    # Limit dataset size sent to AI
                    investigation_df = df.head(100)

                    investigation_prompt = f"""
You are LedgerLens, an AI financial
investigation assistant.

Analyze ONLY the financial evidence
provided below.

Objectives:

1. Identify suspicious financial activity.
2. Identify high and critical risk transactions.
3. Explain important anomalies.
4. Identify unusual financial patterns.
5. Estimate financial impact when supported.
6. Explain possible causes.
7. Provide recommendations for human review.

Rules:

- Do not invent missing information.
- Do not assume fraud without evidence.
- Clearly distinguish anomaly from confirmed fraud.
- Do not authorize financial transactions.
- Recommendations are for human review only.

Return:

Executive Summary:

Key Findings:

High-Risk Activity:

Anomaly Analysis:

Financial Impact:

Possible Causes:

Recommended Actions:

Confidence:

FINANCIAL DATA:

{investigation_df.to_string(index=False)}
"""

                    with st.spinner(
                        "🤖 LedgerLens AI is investigating the financial data..."
                    ):

                        try:

                            response = client.models.generate_content(
                                model="gemini-3.5-flash",
                                contents=investigation_prompt
                            )

                            st.success(
                                "AI investigation completed."
                            )

                            st.text_area(
                                "LedgerLens AI Investigation Report",
                                response.text,
                                height=600
                            )

                        except Exception as e:

                            st.error(
                                f"AI investigation failed: {e}"
                            )


        except Exception as e:

            st.error(
                f"Error processing CSV: {e}"
            )


    # ========================================================
    # PDF
    # ========================================================

    elif file_name.endswith(".pdf"):

        st.success(
            f"✅ {uploaded_file.name} uploaded successfully"
        )

        st.header("📄 PDF Financial Data")

        try:

            extracted_text = extract_pdf_text(
                uploaded_file
            )

            if extracted_text.strip():

                st.success(
                    "PDF text extracted successfully."
                )

                with st.expander(
                    "View extracted PDF data"
                ):

                    st.text_area(
                        "Extracted Content",
                        extracted_text,
                        height=350
                    )


                # --------------------------------------------
                # PDF AI INVESTIGATION
                # --------------------------------------------

                st.divider()

                st.header(
                    "🤖 AI Financial Investigation"
                )

                if st.button(
                    "🚀 Investigate PDF Financial Data",
                    use_container_width=True
                ):

                    if not AI_CONNECTED:

                        st.error(
                            "Gemini AI is not connected."
                        )

                    else:

                        pdf_prompt = f"""
You are LedgerLens, an AI financial
investigation assistant.

Analyze ONLY the financial information
extracted from this PDF.

Identify:

- Suspicious activity
- Financial anomalies
- High-risk information
- Potential financial impact
- Possible causes
- Recommended human review

Do not invent information.
Do not claim fraud without evidence.
Do not authorize transactions.

Return:

Executive Summary:

Key Findings:

Risk Assessment:

Evidence:

Financial Impact:

Possible Causes:

Recommendations:

Confidence:

PDF FINANCIAL DATA:

{extracted_text[:30000]}
"""

                        with st.spinner(
                            "🤖 AI is investigating the PDF..."
                        ):

                            try:

                                response = client.models.generate_content(
                                    model="gemini-3.5-flash",
                                    contents=pdf_prompt
                                )

                                st.success(
                                    "PDF investigation completed."
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

            else:

                st.warning(
                    "No readable text was found in this PDF."
                )

        except Exception as e:

            st.error(
                f"Error extracting PDF: {e}"
            )


# ============================================================
# NO FILE STATE
# ============================================================

else:

    st.info(
        "👆 Upload a CSV or PDF financial dataset to begin."
    )

    st.markdown(
        """
        ### What LedgerLens does

        **📂 Data Upload**  
        Accepts financial CSV and PDF data.

        **📊 Financial Risk Dashboard**  
        Summarizes financial activity and risk.

        **🔍 Anomaly Detection**  
        Identifies unusual transaction values.

        **🚨 Risk Classification**  
        Classifies transactions as Low, Medium, High or Critical.

        **🔎 Single Transaction Investigation**  
        Allows individual transactions to be investigated.

        **🤖 AI Investigation**  
        Generates an evidence-based financial investigation report.

        **👤 Human Review**  
        AI recommendations remain recommendations for human review.
        """
    )
