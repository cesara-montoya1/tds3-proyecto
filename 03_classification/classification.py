import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(
    path: Path, random_state: int = 0
) -> tuple[np.ndarray, pd.Series, object]:
    """
    Load data from a CSV file, separate features and labels,
    and apply standard scaling to features.

    Parameters:
        path (Path): Path to the CSV file
        random_state (int): Seed for the random number generator

    Returns:
        tuple: A tuple containing:
            - X_scaled (np.ndarray): Scaled features.
            - y (pd.Series): Labels.
            - scaler (object): Adapted standard scaler.
    """
    df = pd.read_csv(path)

    # Randomly shuffle the rows of the DataFrame
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Exclude first feature, because it's a data leak for the index of the image
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # Standarize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def load_label_mapping(label_path: Path) -> dict[int, str]:
    """
    Load label mappings from CSV, mapping class numbers to readable names.

    Parameters:
        label_path (Path): Path to the label CSV file

    Returns:
        dict: A dictionary containing:
            - key (int): Class number.
            - value (str): Label name.
    """
    label_df = pd.read_csv(label_path)
    label_mapping = dict(zip(label_df["Class"], label_df["label"]))
    return label_mapping


def load_results(
    results_folder: Path, name: str
) -> tuple[object, object, dict, dict, dict, timedelta] | None:
    """
    Loads previously saved results for a given experiment name from disk.

    This function retrieves train, validation, and test scores, the inference time,
    and the best model saved as part of a previous run. Results are expected to be
    stored in a predefined folder (`RESULTS_FOLDER`) and use a naming convention
    based on the provided `name`.

    Args:
        results_folder (Path): Path to the folder where the files are saved.
        name (str): The identifier for the experiment, used to locate the stored files.

    Returns:
        tuple: A tuple containing:
            - best_estimator (object): The trained model loaded from a pickle file.
            - scaler (object): The adjusted standard scaler.
            - train_scores (dict): Training evaluation metrics.
            - validation_scores (dict): Validation evaluation metrics.
            - test_scores (dict): Test evaluation metrics.
            - inference_time (timedelta): The time taken for inference during the experiment.
        Returns `None` if the files are missing or there is an error in loading.

    Raises:
        OSError: If any of the required files cannot be found or opened.
    """
    try:
        # Load scores
        with open(
            results_folder / f"{name}_train_results.json", "r", encoding="UTF-8"
        ) as f:
            train_scores = json.load(f)
        with open(
            results_folder / f"{name}_validation_results.json", "r", encoding="UTF-8"
        ) as f:
            validation_scores = json.load(f)
        with open(
            results_folder / f"{name}_test_results.json", "r", encoding="UTF-8"
        ) as f:
            test_scores = json.load(f)

        # Load inference time
        with open(
            results_folder / f"{name}_inference_time.txt", "r", encoding="UTF-8"
        ) as f:
            time_str = f.read().strip()
            time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
            inference_time = timedelta(
                hours=time_obj.hour,
                minutes=time_obj.minute,
                seconds=time_obj.second,
                microseconds=time_obj.microsecond,
            )

        # Load the best estimator and standard scaler using pickle
        with open(results_folder / f"{name}_best_estimator.pkl", "rb") as f:
            best_estimator = pickle.load(f)
        with open(results_folder / f"{name}_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        return (
            best_estimator,
            scaler,
            train_scores,
            validation_scores,
            test_scores,
            inference_time,
        )

    except OSError:
        return None


def save_results(
    best_estimator: object,
    scaler: object,
    train_scores: dict,
    validation_scores: dict,
    test_scores: dict,
    inference_time: timedelta,
    results_folder: Path,
    name: str,
) -> None:
    """
    Saves experiment results to disk.

    This function stores train, validation, and test evaluation metrics as JSON files,
    inference time as a text file, the best-trained model and the scaler using pickle.
    The files are named based on the provided experiment name and stored in the
    results folder specified in the `results_folder`.

    Args:
        best_estimator (object): The trained model to be saved.
        scaler (object): The standard scaler to be saved.
        train_scores (dict): Train result scores.
        validation_scores (dict): Validation result scores.
        test_scores (dict): Test result scores.
        inference_time (timedelta): The time taken for inference during the experiment.
        results_folder (Path): Path to the folder where the files are saved.
        name (str): The identifier for the experiment, used to name the stored files.

    Returns:
        None

    Raises:
        OSError: If there is an error in writing any of the files
    """
    # Save metrics to JSON files
    with open(
        results_folder / f"{name}_train_results.json", "w", encoding="UTF-8"
    ) as f:
        json.dump(train_scores, f)
    with open(
        results_folder / f"{name}_validation_results.json", "w", encoding="UTF-8"
    ) as f:
        json.dump(validation_scores, f)
    with open(results_folder / f"{name}_test_results.json", "w", encoding="UTF-8") as f:
        json.dump(test_scores, f)

    # Save inference time
    with open(
        results_folder / f"{name}_inference_time.txt", "w", encoding="UTF-8"
    ) as f:
        f.write(f"{inference_time}")

    # Save the best model and scaler using pickle
    with open(results_folder / f"{name}_best_estimator.pkl", "wb") as f:
        pickle.dump(best_estimator, f)
    with open(results_folder / f"{name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


def grid_search_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    estimator: object,
    param_grid: dict,
    k: int,
    score_metrics: list,
    main_score_metric: callable,
) -> tuple[object, dict, dict, timedelta]:
    """
    Performs a grid search over hyperparameters to find the best model
    and evaluates its performance.

    This function uses GridSearchCV to train a model on the provided
    training data (`X_train` and `y_train`) with a specified estimator,
    parameter grid, and scoring metrics. It returns the best trained model,
    average train and validation scores across cross-validation folds
    for each metric, and the total inference time.

    Args:
        X_train: The feature matrix for training data.
        y_train: The target labels for training data.
        estimator: The estimator to train.
        param_grid: The hyperparameter grid to search over.
        k: The number of cross-validation folds.
        score_metrics: List of metrics to evaluate.
        main_score_metric: The metric to use for model selection.

    Returns:
        tuple: A tuple containing:
            - best_estimator: The model trained with the best combination of hyperparameters.
            - train_scores (dict): A dictionary that averages the train scores for each metric.
            - validation_scores (dict): Same as train scores but for validation.
            - inference_time (timedelta): The total time taken to perform the grid search.

    Raises:
        ValueError: If `X_train` or `y_train` do not match the
        expected dimensions for the estimator.
    """
    # Perform grid search
    grid_search = GridSearchCV(
        estimator,
        param_grid,
        cv=k,
        scoring=score_metrics,
        refit=main_score_metric,
        return_train_score=True,
        verbose=1,
    )

    start_time = datetime.now()
    grid_search.fit(X_train, y_train)
    end_time = datetime.now()

    inference_time = end_time - start_time

    # Extract best estimator and cross-validation results
    best_estimator = grid_search.best_estimator_
    cv_results = grid_search.cv_results_

    # Extract average train and validation scores for each metric
    train_scores = {
        metric: cv_results[f"mean_train_{metric}"].mean()
        for metric in score_metrics.keys()
    }
    validation_scores = {
        metric: cv_results[f"mean_test_{metric}"].mean()
        for metric in score_metrics.keys()
    }

    return best_estimator, train_scores, validation_scores, inference_time


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """
    Evaluate the model's performance using accuracy, precision, recall, and F1 score.

    Args:
        y_true (pd.Series): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        dict: A dictionary containing:
            - key (str): Evaluation metric name.
            - value (str): Evaluation metric score.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }
    return metrics


def print_results(
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    inference_time: timedelta,
) -> None:
    """
    Print evaluation metrics and inference time in a structured and readable format.

    Args:
        train_metrics (dict[str, float]): Metrics for the training set,
                                          where keys are metric names and values are their scores.
        val_metrics (dict[str, float]): Metrics for the validation set,
                                        where keys are metric names and values are their scores.
        test_metrics (dict[str, float]): Metrics for the test set,
                                         where keys are metric names and values are their scores.
        inference_time (timedelta): Total time taken for inference.

    Returns:
        None
    """
    # Print metrics for the training set
    print("\nTrain Set Evaluation Metrics:")
    for metric, value in train_metrics.items():
        # Replace underscores with spaces and capitalize the metric name for better readability
        print(f"  {metric.replace('_', ' ').capitalize()}: {100 * value:.4f}%")

    # Print metrics for the validation set
    print("\nValidation Set Evaluation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric.replace('_', ' ').capitalize()}: {100 * value:.4f}%")

    # Print metrics for the test set
    print("\nTest Set Evaluation Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric.replace('_', ' ').capitalize()}: {100 * value:.4f}%")

    # Print the total inference time
    print("\nInference time:", inference_time)


def plot_metrics(metrics: dict[str, float], name: str) -> None:
    """
    Plot the model evaluation metrics as a bar chart.

    Args:
        metrics (dict[str, float]): A dictionary containing evaluation metrics.
        name (str): Name for the plot.

    Returns:
        None
    """
    plt.figure(figsize=(8, 4))

    for x, y in metrics.items():
        y_ = 100 * y
        plt.bar(x, y_, width=0.5)

    plt.title(f"Model Evaluation Metrics for {name}")
    plt.ylabel("Score in percentage")
    plt.ylim(0, 100)
    plt.show()
    plt.close()


def plot_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, name: str, label_mapping: dict[int, str]
) -> None:
    """
    Plot the confusion matrix for the model's predictions.

    Args:
        y_true (pd.Series): True labels.
        y_pred (np.ndarray): Predicted labels.
        name (str): Name of the plot.
        label_mapping (dict[int, str]): A dictionary mapping class numbers to class names.

    Returns:
        None
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = [label_mapping[i] for i in sorted(label_mapping.keys())]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix for {name}")
    plt.show()
    plt.close()


def plot_grouped_metrics(metrics_dict: dict[str, dict[str, float]]) -> None:
    """
    Plots the evaluation metrics of multiple databases in a grouped bar plot.

    Args:
        metrics_dict (dict[str, dict[str, float]]):
            A dictionary where the keys are database names and
            the values are dictionaries of metrics.

    Returns:
        None
    """
    # Extract database names and metric names
    db_names = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())

    # Create data for plotting
    n_dbs = len(db_names)
    n_metrics = len(metric_names)
    width = 0.1  # Width of each bar

    # Define the x positions for the groups of bars
    x = np.arange(n_dbs)

    # Create a figure and axis
    _, ax = plt.subplots(figsize=(12, 6))

    # Plot each metric for each database
    for i, metric in enumerate(metric_names):
        values = [
            metrics_dict[db][metric] * 100 for db in db_names
        ]  # Convert to percentage
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    # Customize the plot
    ax.set_xlabel("Databases")
    ax.set_ylabel("Score in Percentage")
    ax.set_title("Comparison of Evaluation Metrics Across Databases")
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(db_names)
    ax.legend(title="Metrics", loc="lower center")
    plt.ylim(75, 100)

    plt.show()


def main_pipeline(
    name: str,
    db_path: Path,
    labels_path: Path,
    results_folder: Path,
    random_state: int,
    estimator: object,
    param_grid: dict,
    k: int,
    score_metrics: dict,
    main_score_metric: callable,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """
    Main function to load data, perform SVM classification with grid search,
    cross-validate the model, and evaluate it.

    Args:
        name (str): Name of the database used.
        db_path (Path): Path to the CSV file containing the data.

    Returns:
        None
    """
    print("================================================")
    # Load, standarize data and load class mapping
    X, y, scaler = load_and_preprocess_data(db_path, random_state)
    label_mapping = load_label_mapping(labels_path)

    # Partition data in training and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Flag to know if data was already read from files
    from_files = False

    # Try to load previous results
    print(f"Trying to read previous results for {name}")
    load_result = load_results(
        results_folder,
        name,
    )
    from_files = load_result is not None

    if from_files:
        # Don't load scaler as it was adjusted before using the train data
        (
            best_estimator,
            _,
            train_scores,
            validation_scores,
            test_scores,
            inference_time,
        ) = load_result
    else:
        print(f"Were not able to read previous results, computing new ones for {name}")

        # Perform Grid Search with Cross-Validation
        best_estimator, train_scores, validation_scores, inference_time = (
            grid_search_cv(
                X_train,
                y_train,
                estimator,
                param_grid,
                k,
                score_metrics,
                main_score_metric,
            )
        )

    # Predict on the test set using the best model
    y_pred = best_estimator.predict(X_test)
    test_scores = evaluate_model(y_test, y_pred)
    print_results(train_scores, validation_scores, test_scores, inference_time)
    print("================================================")

    # If not loaded from files save the calculated results
    if not from_files:
        save_results(
            best_estimator,
            scaler,
            train_scores,
            validation_scores,
            test_scores,
            inference_time,
            results_folder,
            name,
        )

    # Plot metrics and confusion matrix
    plot_metrics(test_scores, name)
    plot_confusion_matrix(y_test, y_pred, name, label_mapping)

    return train_scores, validation_scores, test_scores
