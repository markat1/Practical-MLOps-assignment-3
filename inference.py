import os
import joblib
import pandas as pd
import argparse
from preprocessing import DataProcessing


def load_model(model_path):
    return joblib.load(model_path)

def main(args):
    environment = os.getenv("ENVIRONMENT", "local")
    test_data_path = os.path.join(args.inference, "test.csv")
    model_path = "/opt/ml/output/model.joblib"
    output_path = "/opt/ml/output/predictions.csv"

    if environment == "local":
        test_data_path = "data/test/test.csv"
        model_path = "output/model.joblib"
        output_path = "output/predictions.csv"

    # Load model
    # model = load_model(model_path)
    model, label_encoders, scaler, model_columns = joblib.load(model_path)
    print(model_columns)
    print(scaler)

    for column, label_encoder in label_encoders.items():
        X[column] = label_encoder.transform(X[column])

    X = scaler.transform(X)  # Standardize
    predictions = model.predict(X)  # Predict time of delivery

    print(predictions)

    # test_data = pd.read_csv(test_data_path)

    # test_data_processed = preprocess_data(test_data)
    # test_data_processed = preprocess_data(test_data, model_columns=model_columns)

    # predictions = model.predict(test_data_processed)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    pd.DataFrame(predictions, columns=["estimated_time"]).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference",
        type=str,
        help="Path to the inference data",
        default="/opt/ml/input/data/test",
    )
    parser.add_argument("--n-estimators", type=int, default=20)

    args = parser.parse_args()
    main(args)
