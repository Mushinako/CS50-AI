# import csv
import sys
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

MONTHS_LIST = ["Jan", "Feb", "Mar", "Apr", "May", "June",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_DICT = {MONTHS_LIST[i]: i for i in range(12)}


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    df = pd.read_csv(filename, header=0, converters={
        "Administrative": int,
        "Administrative_Duration": float,
        "Informational": int,
        "Informational_Duration": float,
        "ProductRelated": int,
        "ProductRelated_Duration": float,
        "BounceRates": float,
        "ExitRates": float,
        "PageValues": float,
        "SpecialDay": float,
        "Month": lambda m: MONTHS_DICT[m],
        "OperatingSystems": int,
        "Browser": int,
        "Region": int,
        "TrafficType": int,
        "VisitorType": lambda vt: int(vt == "Returning_Visitor"),
        "Weekend": lambda w: int(w == "TRUE"),
        "Revenue": lambda r: int(r == "TRUE"),
    })
    df_labels = df.pop("Revenue")
    return df, df_labels


def train_model(evidences: pd.DataFrame, labels: pd.Series) -> KNeighborsClassifier:
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(evidences, labels)
    return model


def evaluate(labels: pd.Series, predictions: np.ndarray) -> Tuple[float, float]:
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    positive_correct = positive_total = 0
    negative_correct = negative_total = 0
    for i, label in enumerate(labels):
        if label:
            positive_correct += predictions[i]
            positive_total += 1
        else:
            negative_correct += 1-predictions[i]
            negative_total += 1
    return positive_correct/positive_total, negative_correct/negative_total


if __name__ == "__main__":
    main()
