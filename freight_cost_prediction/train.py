import joblib
from pathlib import Path

from data_preprocessing import (
    load_data,
    prepare_data,
    split_data
)

from model_and_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)



def main():

    model_dir= Path("models")
    model_dir.mkdir(exist_ok= True)

    # Load data
    df= load_data(r"C:\Users\Dell\Desktop\ML\Vendor invoice intelligence system\data\inventory.db")

    # Prepare data
    X,y= prepare_data(df)

    # Train Test Split
    X_train, X_test, y_train, y_test = split_data(X,y)
    

    # Train every model
    lr_model= train_linear_regression(X_train, y_train)
    dt_model= train_decision_tree(X_train, y_train)
    rf_model= train_random_forest(X_train, y_train)
    

    # Evaluate every model
    res=[]

    res.append(evaluate_model(lr_model, X_test, y_test, "Linear Regression"))
    res.append(evaluate_model(dt_model, X_test, y_test, "Decision Tree Regression"))
    res.append(evaluate_model(rf_model, X_test, y_test, "Random Forest Regression"))
    

    # Choose the best model
    best_model_info= min(res, key= lambda x: x['RMSE'])
    
    best_model_name= best_model_info['Model']

    best_model = {

        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model

    }[best_model_name]

     
    # Save the best model
    model_path = model_dir / "predict_freight_model.pkl"

    joblib.dump(best_model, model_path)

    print(f"\n Best Model Saved: {best_model_name}")


# Prevents main() from running when this file is imported into another file
if __name__== "__main__":
    main()












