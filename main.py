import os
import platform
import subprocess
import argparse

def run_commands(commands):
    for command in commands:
        subprocess.run(command, shell=True)

if __name__ == "__main__":
    # Set up argparse to accept the model name as a command-line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to train")
    args = parser.parse_args()

    # Check the model argument
    selected_model = args.model

    # Available models from model_dispatcher.py
    available_models = ["logistic_regression", "random_forest", "decision_tree", 
                        "svm", "gradient_boosting", "xgboost"]
    
    if selected_model not in available_models:
        raise ValueError(f"Invalid model. Choose from {available_models}")

    os_type = platform.system()

    if os_type == "Windows":
        # Commands to run on Windows
        commands = [
            f"python ./src/train.py --fold {fold} --model {selected_model}"
            for fold in range(5)
        ]
    elif os_type in ("Linux", "Darwin"):  # Darwin is for macOS
        # Commands to run on Unix-like systems
        commands = [
            f"python3 ./src/train.py --fold {fold} --model {selected_model}"
            for fold in range(5)
        ]
    else:
        raise Exception(f"Unsupported OS: {os_type}")

    run_commands(commands)
