import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

import os
import argparse
import joblib

import config
import model_dispatcher


def run(fold, model):
    # Import the data
    df = pd.read_csv(config.TRAINING_FILE)

    # Split the data into training and testing
    train = df[df.kfold != fold].reset_index(drop=True)
    test = df[df.kfold == fold].reset_index(drop=True)

    # Split the data into features and target
    X_train = train.drop(['id', 'Target', 'kfold'], axis=1)
    X_test = test.drop(['id', 'Target', 'kfold'], axis=1)

    y_train = train.Target.values
    y_test = test.Target.values

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform the training data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Label encode the target
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Initialize the model
    model = model_dispatcher.models[model]

    # Fit the model
    model.fit(X_train, y_train)

    # make predictions
    preds = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    # Save the model
    joblib.dump(model, os.path.join(config.MODEL_OUTPUT, f"model_{fold}.bin"))

if __name__ == '__main__':
    # Initialize the ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # Add the arguments to the parser
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)

    # Read the arguments from the command line
    args = parser.parse_args()

    # Run the fold specified by the command line arguments
    run(fold=args.fold, model=args.model)

