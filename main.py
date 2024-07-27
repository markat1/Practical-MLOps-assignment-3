import subprocess
import os


def main():
    environment = os.getenv("ENVIRONMENT", "local")
    mode = os.getenv("MODE", "train")

    train_path = "/opt/ml/code/train.py"
    inference_path = "/opt/ml/code/inference.py"

    if environment == "local":
        train_path = "train.py"
        inference_path = "inference.py"

    if mode == "train":
        subprocess.run(["python", train_path])
    elif mode == "inference":
        subprocess.run(["python", inference_path])  # Placeholder for inference script


if __name__ == "__main__":
    main()
