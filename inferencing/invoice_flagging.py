import joblib
import pandas as pd

MODEL_PATH = "models/invoice_flagging_model.pkl"


def load_model(model_path=MODEL_PATH):
    """
    Load trained invoice flagging model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model


def predict_invoice_flag(input_data):

    model = load_model()

    input_df = pd.DataFrame(input_data)

    input_df["Predicted_Flag"] = model.predict(input_df)

    return input_df










        


    