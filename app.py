import streamlit as st
from inferencing.predict_freight import predict_freight_cost
from inferencing.invoice_flagging import predict_invoice_flag

# -------------------- Page Config --------------------

st.set_page_config(
    page_title="Vendor Invoice Intelligence",
    page_icon="📊",
    layout="wide"
)

# -------------------- CSS --------------------

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    background-color:#2563EB;
    color:white;
}

[data-testid="metric-container"]{
    border:1px solid #444;
    border-radius:12px;
    padding:15px;
    box-shadow:2px 2px 10px rgba(0,0,0,0.2);
}

h1,h2,h3{
    color:#2E86C1;
}

</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------

st.sidebar.title("📊 Vendor Invoice AI")

choice = st.sidebar.radio(
    "Navigation",
    (
        "🏠 Home",
        "🚚 Freight Cost Prediction",
        "🚩 Invoice Flagging",
        "ℹ️ About"
    )
)

# =====================================================
# HOME
# =====================================================

if choice == "🏠 Home":

    st.title("📊 Vendor Invoice Intelligence System")

    st.markdown("""
### AI Powered Vendor Invoice Analytics

This application helps businesses to:

- 🚚 Predict Freight Cost
- 🚩 Detect High-Risk Invoices
""")

    st.divider()

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("""
        <h4 style='margin-bottom:5px;'>🚚 Freight Model</h4>
        <p style='font-size:22px; color:#4CAF50; font-weight:bold;'>
        Linear Regression
        </p>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <h4 style='margin-bottom:5px;'>🚩 Invoice Model</h4>
        <p style='font-size:22px; color:#2196F3; font-weight:bold;'>
        Random Forest Classifier
        </p>
        """, unsafe_allow_html=True)
    
    st.divider()

    col1,col2 = st.columns(2)

    with col1:

        st.success("""
### 🚚 Freight Cost Prediction

Predict freight cost before processing vendor invoices.

✔ Machine Learning Based

✔ Instant Prediction

✔ Helps estimate logistics cost
""")

    with col2:

        st.error("""
### 🚩 Invoice Flagging

Detect suspicious invoices automatically.

✔ Fraud Detection

✔ Risk Identification

✔ Procurement Analytics
""")

# =====================================================
# FREIGHT
# =====================================================

elif choice == "🚚 Freight Cost Prediction":

    st.title("🚚 Freight Cost Prediction")

    st.caption("Predict expected freight cost for a vendor invoice.")

    col1,col2 = st.columns(2)

    with col1:

        dollars = st.number_input(
            "Invoice Amount ($)",
            min_value=0.0
        )



    st.divider()

    if st.button("🚀 Predict Freight Cost"):

        input_data = {

            "Dollars":[dollars]

        }

        result = predict_freight_cost(input_data)

        predicted = result["Predicted_Freight"][0]

        st.success("Prediction Completed Successfully")

        st.metric(
            "Predicted Freight Cost",
            f"${predicted:,.2f}"
        )

        st.dataframe(result)

# =====================================================
# INVOICE FLAGGING
# =====================================================

elif choice == "🚩 Invoice Flagging":

    st.title("🚩 Invoice Flagging")

    st.caption("Predict whether an invoice is Normal or High Risk.")

    col1,col2 = st.columns(2)

    with col1:

        avg_receiving_days = st.number_input(
            "Average Receiving Days",
            min_value=0.0
        )

        total_purchase_amount = st.number_input(
            "Total Purchase Amount",
            min_value=0.0
        )

        total_quantity = st.number_input(
            "Total Quantity",
            min_value=0
        )

    with col2:

        invoice_quantity = st.number_input(
            "Invoice Quantity",
            min_value=0
        )

        invoice_dollars = st.number_input(
            "Invoice Amount",
            min_value=0.0
        )

    st.divider()

    if st.button("🔍 Analyze Invoice"):

        input_data = {

            "avg_receiving_days":[avg_receiving_days],

            "total_purchase_amount":[total_purchase_amount],

            "total_quantity":[total_quantity],

            "invoice_quantity":[invoice_quantity],

            "invoice_dollars":[invoice_dollars]

        }

        result = predict_invoice_flag(input_data)

        prediction = result["Predicted_Flag"][0]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("🚩 High Risk Invoice Detected")

        else:

            st.success("✅ Normal Invoice")

        st.dataframe(result)

# =====================================================
# ABOUT
# =====================================================

elif choice == "ℹ️ About":
    
    st.title("About This Project")

    st.markdown("""
    ## Vendor Invoice Intelligence System
    
    ### Project Objective
    
    This project uses Machine Learning to automate vendor invoice analysis.
    
    ---
    
    ### Modules
    
    🚚 **Freight Cost Prediction**
    
    Predicts expected freight cost using Linear Regression.
    
    ---
    
    🚩 **Invoice Flagging**
    
    Detects suspicious invoices using Random Forest Classification.
    
    ---
    
    ### Technologies Used
    
    - Python
    - SQL
    - Pandas
    - Scikit-Learn
    - Streamlit
    - Joblib
    
    ---
    
    ### Machine Learning Models
    
    - Linear Regression
    - Random Forest Classifier
    
    ---
    
    ### Developed By
    
    **Antima Goyal**
    """)