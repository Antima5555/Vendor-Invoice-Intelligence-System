from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def train_linear_regression(X_train,y_train):
    model= LinearRegression()
    model.fit(X_train,y_train)
    return model


def train_decision_tree(X_train,y_train):
    model= DecisionTreeRegressor(max_depth= 5)
    model.fit(X_train,y_train)
    return model


def train_random_forest(X_train,y_train):
    model= RandomForestRegressor(max_depth= 5, random_state=42)
    model.fit(X_train,y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name:str):

    y_pred= model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name}")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print(f"R2 Score: {r2*100:.2f}%")

    return {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2 * 100, 2)
    }















