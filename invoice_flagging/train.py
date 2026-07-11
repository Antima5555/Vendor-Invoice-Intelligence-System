from preprocessing import (load_data, apply_labels, split_data,scale_features)
from model_and_evaluation import (train_random_forest,evaluate_model)
import joblib

features = ["avg_receiving_days", "total_purchase_amount","total_quantity","invoice_quantity","invoice_dollars"]
target = "flag_invoice"



def main():

    # Load Data
    df= load_data(r"C:\Users\Dell\Desktop\ML\Vendor invoice intelligence system\data\inventory.db")

    # Create Target Variable
    df = apply_labels(df)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = split_data(df,features,target)
    

    # Feature Scaling
    X_train_scaled, X_test_scaled = scale_features(X_train,X_test)

    
    # Train Model
    model = train_random_forest(X_train_scaled,y_train)

    # Evaluate Model
    evaluate_model(model, X_test_scaled, y_test)

    # Save Model
    joblib.dump(model,"models/invoice_flagging_model.pkl")

    print("\nModel saved successfully!")


if __name__ == "__main__":
    main()