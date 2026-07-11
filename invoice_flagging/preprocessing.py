import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_data(db_path):

    conn = sqlite3.connect(db_path)

    query = """
     WITH purchase_agg AS (
     SELECT
        PONumber,
        COUNT(DISTINCT Brand) AS total_brands,
        SUM(Quantity) AS total_quantity,
        SUM(Dollars) AS total_purchase_amount,
        AVG(julianday(ReceivingDate) - julianday(PODate)) AS avg_receiving_days
     FROM purchases
     GROUP BY PONumber
    )

     SELECT
       vi.PONumber,
       vi.Quantity AS invoice_quantity,
       vi.Dollars AS invoice_dollars,
       vi.Freight,

      (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS po_to_invoice_days,
      (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS payment_days,

       pa.total_brands,
       pa.total_quantity,
       pa.total_purchase_amount,
       pa.avg_receiving_days

     FROM vendor_invoice AS vi

     LEFT JOIN purchase_agg AS pa

     ON vi.PONumber = pa.PONumber
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# Create Target
def create_invoice_risk_label(row):

    if abs(row["invoice_dollars"] - row["total_purchase_amount"]) > 5:
        return 1
    if row["avg_receiving_days"] > 10:
        return 1

    return 0


# Apply on dataframe
def apply_labels(df):
    
    df["flag_invoice"] = df.apply(create_invoice_risk_label,axis=1)
    return df


# Train Test Split

def split_data(df, feature_columns,target_column):

    X = df[feature_columns]

    y = df[target_column]

    return train_test_split(X, y, test_size=0.20, random_state=42,stratify=y)

    

# Feature Scaling

def scale_features(X_train,X_test,scaler_path="models/scaler.pkl"):

    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled









