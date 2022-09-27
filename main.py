""" This file contains driver code for the rule generation algorithm
    It processes datasets in the following steps:
    1. Set parameters for data processing (ex: which dataset to use, how many rules to generate, etc.)
    2. Binarize the data...
        a. Convert numerical features to categorical
        b. For each categorical feature, record "True" if it holds for a certain datapoint, "False" otherwise
        c. Record the results in an external Pickle file so that this process only needs to be done once per dataset
    3. Generate and process possible DNF models with bestModels.py
    4. Filter out similar models and record the results in an external Excel file """

from loadData import load_data
from bestModels import find_best_model
from beautifyFormulas import beautify_formulas

import numpy as np
import pandas as pd
import fastmetrics
import os.path
from sklearn.model_selection import train_test_split
from os import cpu_count

# Dataset settings
INPUT_FILE = "DivideBy30"
TARGET_NAME = "Div By 30"
FILE_EXT = "xlsx"
DROP_FEATURE = ["Number"]

# Model settings
SUBSET_SIZE = 2
NUM_MODELS = 1000

# Binarize settings
DELETE_NANS = True
NUM_BINS = 20

# Parallel settings (-1 == sequential)
NUM_THREADS = int(max(cpu_count()*.9, 1))

# Filter features to reduce runtime (-1 = no filter)
NUM_FEATURES = 100

# Similarity Comparison Metric
# Option 1: Jacquard Score ("JAC_SCORE") with Threshold
# Option 2: Parent Similarity ("PARENT_SIMILARITY") for models with the same features
METRIC = "PARENT_SIMILARITY"
MIN_JAC_SCORE = .9  # Threshold for Jacquard Score
SIMILAR_FEATURES = 2  # Number of equal features necessary for similarity in Parent Similarity
SIMILAR_EXACT = True  # Should exactly equal models be filtered, even if they don't pass the SIMILAR_FEATURES threshold?


# Return True if two models are deemed "similar" by METRIC, False otherwise
# Keynote: Model format = (F1 Score, Features, Expr, Testing Results, Training Results)
def compare_model_similarity(model1, model2):
    if METRIC == "JAC_SCORE":
        return fastmetrics.fast_jaccard_score(model1[-1], model2[-1]) > MIN_JAC_SCORE

    elif METRIC == "PARENT_SIMILARITY":
        model1_features = []
        model2_features = []

        for i in range(SUBSET_SIZE):
            if f"df[columns[{i}]]" in model1[2]:
                model1_features.append(model1[1][i])

            if f"df[columns[{i}]]" in model2[2]:
                model2_features.append(model2[1][i])

        cnt = 0

        for feature in model1_features:
            if feature in model2_features:
                cnt += 1

        # If their similarity exceeds the threshold, or they're exactly the same
        return cnt >= SIMILAR_FEATURES or (SIMILAR_EXACT and model1_features == model2_features)

    # Default for no metric specified = False
    else:
        return False


def main():
    # Check that the input file exists
    file = INPUT_FILE + "." + FILE_EXT

    if not os.path.exists(file):
        print("Input file not found!")
        return

    print("Loading data...")
    pickle_file = INPUT_FILE + "Binarized.pkl"

    # If we've already binarized this data...
    if os.path.exists(pickle_file):
        print("Data was loaded from Pickle")
        df = pd.read_pickle(pickle_file)

    else:
        print('Binarizing data...')
        df = load_data(INPUT_FILE, FILE_EXT, TARGET_NAME)

        # Drop unnecessary features
        for feature in DROP_FEATURE:
            df.drop(feature, axis=1, inplace=True)

        # Binarize the dataset and record the results
        for col in df.columns:
            print(col)
            vals = set(df[col].tolist())

            if vals != {False, True}:
                # Bin continuous variables
                if len(vals) > NUM_BINS and (isinstance(next(iter(vals)), int) or
                                             isinstance(next(iter(vals)), float)):
                    df[col] = pd.cut(df[col], NUM_BINS)

                # One hot encode the categorical variables
                df = pd.get_dummies(df, [col+"="], "", columns=[col])

        # Delete all null columns
        if DELETE_NANS:
            for col in df.columns:
                if 'nan' in col or 'NULL' in col:
                    df.drop(col, axis=1, inplace=True)

        df.to_pickle(pickle_file)

    # Generate training and testing data
    y_true = df['Target']
    df.drop('Target', axis=1, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(df, y_true, test_size=0.2)

    if NUM_FEATURES != -1:
        print("Filtering features...")
        best_formulas = find_best_model(x_train, x_test, y_train, 1,  False, NUM_THREADS, NUM_MODELS,
                                        METRIC == "PARENT_SIMILARITY" and SIMILAR_FEATURES)
        best_features = [formula[1][0] for formula in best_formulas[:NUM_FEATURES]]

        for col in df.columns:
            if col not in best_features and "~" + col not in best_features:
                x_train.drop(col, axis=1, inplace=True)
                x_test.drop(col, axis=1, inplace=True)

    print('Begin training...')
    best_formulas = find_best_model(x_train, x_test, y_train, SUBSET_SIZE, NUM_THREADS != -1, NUM_THREADS, NUM_MODELS,
                                    METRIC == "PARENT_SIMILARITY" and SIMILAR_FEATURES)

    # Sort by F1 score, then shortest expression
    best_formulas.sort(key=lambda x: (-x[0], len(x[2])))

    # Only consider the top "n" (NUM_MODELS) models
    best_formulas = best_formulas[:NUM_MODELS]

    # Filter similar models
    i = 0

    while i < len(best_formulas):
        for j in range(len(best_formulas)-1, i, -1):
            if compare_model_similarity(best_formulas[i], best_formulas[j]):
                del best_formulas[j]
        i += 1

    beautiful_forms = beautify_formulas(best_formulas, np.array(y_test), SUBSET_SIZE)
    models = pd.DataFrame(data=beautiful_forms, columns=['F1', 'TPR', 'FPR', 'FNR', 'TNR', 'Precision', 'Recall',
                                                         'ROC AUC', 'Accuracy', 'Simple DNF'])
    models.to_excel(INPUT_FILE + "Results.xlsx", index=False, freeze_panes=(1, 1))


if __name__ == '__main__':
    main()
