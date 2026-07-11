# 📊 Vendor Invoice Intelligence System

An end-to-end Machine Learning application that automates vendor invoice analysis by predicting freight costs and identifying high-risk invoices. The project integrates SQL, Python, Machine Learning, and Streamlit into a complete business solution.

---

## 📌 Project Overview

The system consists of two independent Machine Learning modules:

### 🚚 Freight Cost Prediction
Predicts the expected freight cost based on invoice information.

### 🚩 Invoice Flagging
Identifies whether an invoice is Normal or High Risk using a classification model.

The application is deployed as an interactive Streamlit web application.

---

## 🎯 Business Problem

Organizations process thousands of vendor invoices every month. Manual verification of freight costs and invoice validation is time-consuming and prone to errors.

This project helps by:

- Predicting freight costs automatically
- Detecting potentially suspicious invoices
- Reducing manual verification effort
- Supporting procurement decision-making

---

## 🛠️ Technologies Used

- Python
- SQL (SQLite)
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- VS Code
- Jupyter Notebook

---

## 📈 Machine Learning Models

### Freight Cost Prediction

- Linear Regression

Evaluation Metrics

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

### Invoice Flagging

- Random Forest Classifier

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 📁 Project Structure

```text
Vendor_Invoice_Intelligence_System/

│

├── data/

├── freight_cost_prediction/

├── invoice_flagging/

├── inferencing/

├── models/

├── app.py

└── notebooks
```

---

## 🚀 How to Run

### Launch the application

```bash
streamlit run app.py
```

---

## 📈 Business Impact

- Predicts freight costs before invoice approval
- Detects high-risk invoices automatically
- Reduces manual processing time
- Improves procurement efficiency
- Supports data-driven business decisions

---

## 👩‍💻 Developed By

**Antima Goyal**

Aspiring Data Analyst | Business Analytics Enthusiast

