import streamlit as st
import pandas as pd
from pypdf import PdfReader
from google import genai


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LedgerLens",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# GEMINI CLIENT
# ============================================================

try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )
except Exception:
    client = None


# ============================================================
# HEADER
# ============================================================

st.title("💰 LedgerLens")
st.subheader("AI Finance Controller")

st.write(
    "AI-powered financial risk intelligence and investigation platform."
)

st.divider()


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("📂 Upload Financial Data")

uploaded_file = st.file_uploader(
    "Upload a CSV or PDF financial report",
    type=["csv", "pdf"]
)


# ============================================================
# CSV PROCESSING
# ============================================================

if uploaded_file is not None and uploaded_file.name.lower().endswith(".csv"):

    try:

        # ----------------------------------------------------
        # READ CSV
        # ----------------------------------------------------

        df = pd.read_csv(uploaded_file)

        st.success(
            f"Successfully loaded: {uploaded_file.name}"
        )

        # ----------------------------------------------------
        # BASIC CLEANING
        # ----------------------------------------------------

        if "amount" in df.columns:

            df["amount"] = pd.to_numeric(
                df["amount"],
                errors="coerce"
            )

        elif "amount_inr" in df.columns:

            df["amount_inr"] = pd.to_numeric(
                df["amount_inr"],
                errors="coerce"
            )

            df["amount"] = df["amount_inr"]

        else:

            st.error(
                "This CSV does not contain an amount column."
            )

            st.stop()


        # ----------------------------------------------------
        # DATE DETECTION
        # ----------------------------------------------------

        date_column = None

        possible_date_columns = [
            "date",
            "payment_date",
            "transaction_date",
            "settlement_date"
        ]

        for col in possible_date_columns:

            if col in df.columns:
                date_column = col
                df[col] = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )
                break


        # ----------------------------------------------------
        # ANOMALY DETECTION
        # ----------------------------------------------------

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
                    (df["amount"] - mean_amount).abs()
                    / std_amount
                ) * 25
            )

            df["anomaly_score"] = df[
                "anomaly_score"
            ].clip(0, 100)


        # ----------------------------------------------------
        # RISK CLASSIFICATION
        # ----------------------------------------------------

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


        # ====================================================
        # DASHBOARD HEADER
        # ====================================================

        st.divider()

        st.header("📊 Financial Risk Dashboard")

        st.caption(
            f"Dataset: {uploaded_file.name}"
        )


        # ====================================================
        # KPI CARDS
        # ====================================================

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

        total_amount = df["amount"].sum()

        high_risk_amount = df.loc[
            df["risk_level"].isin(
                ["HIGH", "CRITICAL"]
            ),
            "amount"
        ].sum()


        # ----------------------------------------------------
        # NATIVE STREAMLIT KPI CARDS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Records",
                f"{total_transactions:,}"
            )

        with col2:

            st.metric(
                "🔴 High / Critical",
                f"{high_risk_count:,}"
            )

        with col3:

            st.metric(
                "🟠 Medium Risk",
                f"{medium_risk_count:,}"
            )

        with col4:

            st.metric(
                "🟢 Low Risk",
                f"{low_risk_count:,}"
            )


        # ----------------------------------------------------
        # SECOND KPI ROW
        # ----------------------------------------------------

        st.write("")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "💰 Total Amount",
                f"₹{total_amount:,.2f}"
            )

        with col2:

            st.metric(
                "⚠️ High-Risk Amount",
                f"₹{high_risk_amount:,.2f}"
            )

        with col3:

            if total_amount != 0:

                risk_percentage = (
                    high_risk_amount
                    / total_amount
                ) * 100

            else:

                risk_percentage = 0

            st.metric(
                "Risk Exposure",
                f"{risk_percentage:.1f}%"
            )


        st.divider()


        # ====================================================
        # CHARTS
        # ====================================================

        st.header("📈 Financial Analytics")


        # ----------------------------------------------------
        # RISK DISTRIBUTION
        # ----------------------------------------------------

        chart_col1, chart_col2 = st.columns(2)


        with chart_col1:

            st.subheader("Risk Distribution")

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

            risk_chart = pd.DataFrame(
                {
                    "Transactions": risk_counts
                }
            )

            st.bar_chart(
                risk_chart,
                use_container_width=True
            )


        # ----------------------------------------------------
        # AMOUNT BY RISK
        # ----------------------------------------------------

        with chart_col2:

            st.subheader("Financial Exposure by Risk")

            risk_amount = (
                df.groupby("risk_level")["amount"]
                .sum()
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

            amount_chart = pd.DataFrame(
                {
                    "Amount": risk_amount
                }
            )

            st.bar_chart(
                amount_chart,
                use_container_width=True
            )


        # ====================================================
        # DATE TREND
        # ====================================================

        if date_column is not None:

            st.subheader("📅 Transaction Trend")

            daily_data = (
                df.dropna(subset=[date_column])
                .groupby(date_column)
                .size()
                .rename("Transactions")
            )

            daily_data = daily_data.sort_index()

            if len(daily_data) > 1:

                st.line_chart(
                    daily_data,
                    use_container_width=True
                )

            else:

                st.info(
                    "Not enough date information to display a trend."
                )


        # ====================================================
        # CATEGORY ANALYSIS
        # ====================================================

        category_column = None

        for col in [
            "category",
            "Category",
            "expense_category"
        ]:

            if col in df.columns:
                category_column = col
                break


        if category_column is not None:

            st.subheader("🗂️ Financial Activity by Category")

            category_data = (
                df.groupby(category_column)["amount"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            st.bar_chart(
                category_data,
                use_container_width=True
            )


        st.divider()


        # ====================================================
        # HIGH-RISK TRANSACTIONS
        # ====================================================

        st.header("🚨 High-Risk Transactions")

        high_risk = df[
            df["risk_level"].isin(
                ["HIGH", "CRITICAL"]
            )
        ].sort_values(
            "anomaly_score",
            ascending=False
        )


        if len(high_risk) > 0:

            st.warning(
                f"{len(high_risk)} high or critical "
                "transactions require human review."
            )

            display_columns = [
                "transaction_id",
                "order_id",
                "customer_id",
                "date",
                "payment_date",
                "amount",
                "payment_status",
                "anomaly_score",
                "risk_level"
            ]

            display_columns = [
                col
                for col in display_columns
                if col in high_risk.columns
            ]

            st.dataframe(
                high_risk[display_columns],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "No high-risk transactions detected."
            )


        # ====================================================
        # INDIVIDUAL TRANSACTION INVESTIGATION
        # ====================================================

        st.divider()

        st.header("🔎 Examine Individual Transaction")

        if len(df) > 0:

            transaction_options = list(
                range(len(df))
            )

            selected_index = st.selectbox(
                "Select a transaction to examine",
                transaction_options,
                format_func=lambda x: (
                    str(
                        df.iloc[x].get(
                            "transaction_id",
                            f"Transaction {x + 1}"
                        )
                    )
                )
            )

            selected_transaction = df.iloc[
                selected_index
            ]


            # ------------------------------------------------
            # TRANSACTION DETAILS
            # ------------------------------------------------

            st.subheader("Transaction Details")

            detail_columns = st.columns(4)

            important_fields = [
                ("Transaction ID", "transaction_id"),
                ("Amount", "amount"),
                ("Risk Level", "risk_level"),
                ("Anomaly Score", "anomaly_score")
            ]

            for column, (label, field) in zip(
                detail_columns,
                important_fields
            ):

                with column:

                    if field in selected_transaction.index:

                        value = selected_transaction[field]

                        if field == "amount":

                            st.metric(
                                label,
                                f"₹{value:,.2f}"
                                if pd.notna(value)
                                else "N/A"
                            )

                        elif field == "anomaly_score":

                            st.metric(
                                label,
                                f"{value:.2f}"
                                if pd.notna(value)
                                else "N/A"
                            )

                        else:

                            st.metric(
                                label,
                                str(value)
                            )


            # ------------------------------------------------
            # FULL RECORD
            # ------------------------------------------------

            with st.expander(
                "View complete transaction record"
            ):

                st.dataframe(
                    selected_transaction
                    .to_frame("Value"),
                    use_container_width=True
                )


            # ------------------------------------------------
            # AI INVESTIGATION
            # ------------------------------------------------

            st.subheader(
                "🤖 AI Transaction Investigation"
            )

            if client is None:

                st.error(
                    "Gemini API key is not configured."
                )

            else:

                if st.button(
                    "Investigate Selected Transaction",
                    type="primary"
                ):

                    transaction_text = (
                        selected_transaction
                        .to_string()
                    )

                    transaction_prompt = f"""
You are LedgerLens, an AI financial investigation assistant.

Analyze ONLY the transaction evidence provided below.

Rules:
1. Identify suspicious or abnormal characteristics.
2. Explain the evidence clearly.
3. Mention the financial impact when supported.
4. Do not invent missing information.
5. If evidence is insufficient, explicitly say so.
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

TRANSACTION EVIDENCE:

{transaction_text}
"""

                    with st.spinner(
                        "AI is examining the transaction..."
                    ):

                        try:

                            response = client.models.generate_content(
                                model="gemini-3.5-flash",
                                contents=transaction_prompt
                            )

                            st.success(
                                "Transaction investigation completed."
                            )

                            st.text_area(
                                "AI Investigation Report",
                                response.text,
                                height=400
                            )

                        except Exception as e:

                            st.error(
                                f"AI investigation failed: {e}"
                            )


        # ====================================================
        # COMPLETE DATASET
        # ====================================================

        st.divider()

        st.header("📋 Complete Analyzed Dataset")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # DOWNLOAD
        # ====================================================

        csv_data = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Analyzed CSV",
            data=csv_data,
            file_name="ledgerlens_analyzed_data.csv",
            mime="text/csv"
        )


    except Exception as e:

        st.error(
            f"Error processing CSV: {e}"
        )


# ============================================================
# PDF PROCESSING
# ============================================================

elif (
    uploaded_file is not None
    and uploaded_file.name.lower().endswith(".pdf")
):

    st.success(
        f"Successfully loaded: {uploaded_file.name}"
    )

    st.header("📄 PDF Financial Report")

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


        # ----------------------------------------------------
        # DISPLAY EXTRACTED TEXT
        # ----------------------------------------------------

        if extracted_text.strip():

            st.success(
                "PDF text extracted successfully."
            )

            with st.expander(
                "View extracted PDF content"
            ):

                st.text_area(
                    "PDF Content",
                    extracted_text,
                    height=400
                )


            # ------------------------------------------------
            # AI PDF INVESTIGATION
            # ------------------------------------------------

            st.header(
                "🤖 AI Financial Investigation"
            )

            if client is None:

                st.error(
                    "Gemini API key is not configured."
                )

            else:

                if st.button(
                    "Investigate Financial Data",
                    type="primary"
                ):

                    investigation_prompt = f"""
You are LedgerLens, an AI financial investigation assistant.

Analyze ONLY the financial evidence provided below.

Rules:
1. Identify suspicious or abnormal financial activity.
2. Explain the evidence clearly.
3. Calculate or mention financial impact when supported.
4. Do not invent missing information.
5. If evidence is insufficient, explicitly say so.
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
                                "Investigation completed."
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


# ============================================================
# NO FILE
# ============================================================

else:

    st.info(
        "Upload a CSV or PDF financial report to begin analysis."
    )

    st.write(
        "LedgerLens will automatically analyze the uploaded "
        "financial data and generate a risk dashboard."
    )
