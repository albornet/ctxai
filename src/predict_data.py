# Basics
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import config
except:
    from . import config
logger = config.CTxAILogger("INFO")

# Models and data processing
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    r2_score,
    classification_report
)

# Utils
from imblearn.over_sampling import SMOTE
from smogn import smoter
from tqdm import tqdm
from collections import defaultdict, Counter
from predict_utils import (
    select_normalizers,
    get_ct_target,
    find_best_model,
    suggest_model,
    compute_medoid,
    post_process_ct_targets,
)


def main():
    """ Train an scikit-learn model on a CT-level classification task using
        cluster affordance of its eligibility criteria as input 
    """
    # Get configuration and go through all embedding models
    cfg = config.get_config()
    for embed_model_id in cfg["EMBEDDING_MODEL_ID_MAP"].keys():
        
        # Build dataset and create splits
        result_dir = os.path.join(cfg["RESULT_DIR"], embed_model_id)
        X, y = extract_prediction_dataset(result_dir)        
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            X, y, test_size=0.3, random_state=cfg["RANDOM_STATE"],
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test, test_size=0.5, random_state=cfg["RANDOM_STATE"],
        )
        if cfg["BALANCE_PREDICTION_DATA"]:
            X_train, y_train = balance_prediction_data(X_train, y_train)
        
        # Normalize input features and output targets
        normalizer_X, normalizer_y = select_normalizers()
        X_train = normalizer_X.fit_transform(X_train)
        X_val = normalizer_X.fit_transform(X_val)
        X_test = normalizer_X.transform(X_test)
        
        y_train = normalizer_y.fit_transform(y_train)
        y_val = normalizer_y.transform(y_val)
        y_test = normalizer_y.transform(y_test)
        
        # Find best hyper-parameters and fit final regression model
        best_params = find_best_model(X_train, y_train, X_val, y_val)
        best_model = suggest_model(model_params=best_params)
        best_model.fit(X_train, y_train)
        
        # Test best model
        test_model(result_dir, best_model, X_test, y_test, normalizer_y)
        
        # FOR NOW ONLY DOING ONE MODEL
        break


def extract_prediction_dataset(result_dir: str):
    """ Extract a dataset paring EC cluster affordances for any clinical trial
        to the corresponding target defined by target_type

    Args:
        result_dir (str): directory containing cluster results
        
    Returns:
        (tuple[np.ndarray, np.ndarray]): input feature and target arrays
    """
    # Dataset for debbugging
    cfg = config.get_config()
    if cfg["PREDICTOR_TARGET_TYPE"] == "debug":
        match cfg["PREDICTOR_TASK_TYPE"]:
        
            case "regression":
                return make_regression(
                    n_samples=1000, n_features=100, n_informative=10, n_targets=1,
                )
                
            case "classification":
                return make_classification(
                    n_samples=1000, n_features=20, n_informative=4, n_classes=4,
                )
    
    # Collect raw embeddings and cluster output data
    raw_embedding_path = result_dir.replace("results/", "processed/embeddings_") + ".pt"
    raw_embeddings = torch.load(raw_embedding_path).numpy()
    cluster_result_path = os.path.join(result_dir, "ec_clustering.json")
    with open(cluster_result_path, "r") as f:
        cluster_output_data = json.load(f)
    
    # Pre-process each clinical trial data
    cluster_dict = defaultdict(list)
    red_embedding_dict = defaultdict(list)
    raw_embedding_dict = defaultdict(list)
    raw_medoid_dict = defaultdict(list)
    red_medoid_dict = defaultdict(list)
    for cluster_instance in tqdm(
        iterable=cluster_output_data["cluster_instances"],
        total=len(cluster_output_data["cluster_instances"]),
        desc="Building input features from eligibility criteria",
    ):
        cluster_id = cluster_instance["cluster_id"]
        ec_list = cluster_instance["ec_list"]
        ec_ids = [ec["member_id"] for ec in ec_list]
        raw_cluster_medoid = compute_medoid(raw_embeddings[ec_ids])
        red_cluster_medoid = cluster_instance["medoid"]
        for ec, ec_id in zip(ec_list, ec_ids):
            ct_path = ec["ct_id"]
            raw_embedding = raw_embeddings[ec_id]
            red_embedding = ec["reduced_embedding"]
            cluster_dict[ct_path].append(cluster_id)
            red_embedding_dict[ct_path].append(red_embedding)
            raw_embedding_dict[ct_path].append(raw_embedding)
            raw_medoid_dict[ct_path].append(raw_cluster_medoid)
            red_medoid_dict[ct_path].append(red_cluster_medoid)
            
    # Initialize final dataset
    n_samples_max = len(cluster_dict)
    n_clusters = len(cluster_output_data["cluster_instances"])
    raw_embedding_dim = len(raw_embedding)  # last one loaded
    red_embedding_dim = len(red_embedding)  # last one loaded
    raw_complete_dim = raw_embedding_dim * n_clusters
    red_complete_dim = red_embedding_dim * n_clusters
    output_targets = np.empty((n_samples_max,), dtype=object)
    cluster_features = np.zeros((n_samples_max, n_clusters))
    raw_embedding_features = np.zeros((n_samples_max, raw_embedding_dim))
    red_embedding_features = np.zeros((n_samples_max, red_embedding_dim))
    raw_medoid_features = np.zeros((n_samples_max, raw_embedding_dim))
    red_medoid_features = np.zeros((n_samples_max, red_embedding_dim))
    
    # Extract feature target pairs
    for row_id, (
            (ct_path_1, cluster_data),
            (ct_path_2, raw_embedding_data),
            (ct_path_3, red_embedding_data),
            (ct_path_4, raw_medoid_data),
            (ct_path_5, red_medoid_data),
        ) in tqdm(
        iterable=enumerate(zip(
            cluster_dict.items(),
            raw_embedding_dict.items(),
            red_embedding_dict.items(),
            raw_medoid_dict.items(),
            red_medoid_dict.items(),
        )),
        total=len(cluster_dict),
        desc="Extracting feature-target pairs",
    ):
        # Load clinical trial data
        assert ct_path_1 == ct_path_2 == ct_path_3 == ct_path_4 == ct_path_5
        with open(ct_path_1, "r") as f:
            ct_data = json.load(f)
        
        # Extract target (or None if not found)
        ct_target = get_ct_target(ct_data)
        output_targets[row_id] = ct_target
        
        # Build cluster input feature vector from EC cluster affordances
        cluster_counts = Counter(cluster_data)
        cluster_col_ids = [n_clusters - 1 if c == -1 else c for c in cluster_counts.keys()]
        cluster_features[row_id][cluster_col_ids] = list(cluster_counts.values())
        
        # Build raw and reduced embedding feature vectors
        raw_embedding_features[row_id] = np.mean(raw_embedding_data, axis=0)
        red_embedding_features[row_id] = np.mean(red_embedding_data, axis=0)
        
        # Build raw and reduced medoid feature vectors
        raw_medoid_features[row_id] = np.mean(raw_medoid_data, axis=0)
        red_medoid_features[row_id] = np.mean(red_medoid_data, axis=0)
    
    # Build complete vectors that include both cluster and embedding information
    raw_complete_features = np.hstack((raw_embedding_features, cluster_features))
    red_complete_features = np.hstack((red_embedding_features, cluster_features))
    
    # Build final dataset (removing samples with no label found)
    mask = np.vectorize(lambda x: x is not None)(output_targets)
    output_targets = output_targets[mask]
    output_targets = post_process_ct_targets(output_targets)
    cluster_features = cluster_features[mask]
    raw_embedding_features = raw_embedding_features[mask]
    red_embedding_features = red_embedding_features[mask]
    raw_medoid_features = raw_medoid_features[mask]
    red_medoid_features = red_medoid_features[mask]
    raw_complete_features = raw_complete_features[mask]
    red_complete_features = red_complete_features[mask]
    
    # Select input feature type
    match cfg["PREDICTOR_INPUT_TYPE"]:
        case "cluster_ids":
            input_features = cluster_features
        case "raw_embeddings":
            input_features = raw_embedding_features
        case "reduced_embeddings":
            input_features = red_embedding_features
        case "raw_medoids":
            input_features = raw_medoid_features
        case "reduced_medoids":
            input_features = red_medoid_features
        case "raw_completes":
            input_features = raw_complete_features
        case "reduced_completes":
            input_features = red_complete_features
        case "random":
            input_features = np.random.rand(*cluster_features.shape)
        case _:
            raise ValueError("Invalid predictor input type")
    
    # Print task information
    print("Using %s model to predict %s from %s input features of shape = %s" % (
        cfg["PREDICTOR_MODEL_TYPE"],
        cfg["PREDICTOR_TARGET_TYPE"],
        cfg["PREDICTOR_INPUT_TYPE"],
        input_features.shape,
    ))
    
    # Adjust data type for regression
    if cfg["PREDICTOR_TASK_TYPE"] == "regression":
        output_targets = output_targets.astype(np.float32)
    else:
        num_classes = len(list(set(output_targets)))
        print("Number of classes: %s" % num_classes)
    
    return input_features, output_targets


def balance_prediction_data(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray]:
    """ Balance data given output targets, handling classification or regression

    Args:
        X (np.ndarray): input features of the training data
        y (np.ndarray): output targets of the training data

    Returns:
        tuple[np.ndarray]: balanced inpute features and output targets
    """
    cfg = config.get_config()
    match cfg["PREDICTOR_TASK_TYPE"]:
        
        case "regression":
            X_y = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            X_y["target"] = y
            X_y_resampled = smoter(data=X_y, y="target")
            X_resampled = X_y_resampled.drop(columns=["target"]).values
            y_resampled = X_y_resampled["target"].values
            
        case "classification":
            smote = SMOTE(random_state=cfg["RANDOM_STATE"])
            X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled
    

def test_model(
    result_dir: str,
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    normalizer_y: Pipeline|StandardScaler,
):
    """ Generate a csv file containing model prediction and true target values
        for each sample of the testing dataset

    Args:
        result_dir (str): directory in which results are saved
        model (BaseEstimator): model trained with the training dataset
        X_test (np.ndarray): input features of the testing dataset
        y_test (np.ndarray): output targets of the testing dataset
        normalizer_y (Pipeline|StandardScaler): normalizer used during training
    """
    # Predict the test set using the best model
    y_pred = model.predict(X_test)
    
    # Inverse transform the predictions and actual values to compare them in their original scale
    y_test = normalizer_y.inverse_transform(y_test)
    y_pred = normalizer_y.inverse_transform(y_pred)
    
    # Write actual and predicted values to a csv file
    results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    csv_path = os.path.join(result_dir, "predict_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Wite results to a png file (scatter plot comparing predictions and targets)
    cfg = config.get_config()
    plot_path = os.path.join(result_dir, "predict_results.png")
    match cfg["PREDICTOR_TASK_TYPE"]:
        
        case "regression":
            plot_regression_results(plot_path, y_test, y_pred)
        
        case "classification":
            class_names = normalizer_y.classes_
            plot_classification_results(plot_path, class_names, y_test, y_pred)
    

def plot_regression_results(
    plot_path: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> plt.Figure:
    """ Create a scatter plot of actual vs predicted values

    Args:
        plot_path (str): path to which the plot will be saved
        y_test (np.ndarray): true labels
        y_pred (np.ndarray): predicted labels
    """
    r_squared = r2_score(y_test, y_pred)
    print(f"R^2 score: {r_squared}")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
    plt.title("Actual vs Predicted Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.xscale("log"); plt.yscale("log")
    plt.grid(True)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "red")
    plt.text(
        min(y_test), max(y_pred), f"R^2 = {r_squared:.2f}",
        fontsize=12, bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.savefig(plot_path)


def plot_classification_results(
    plot_path: str,
    class_names: np.ndarray[str],
    y_test: np.ndarray,
    y_pred: np.ndarray,
):
    """ Plots a confusion matrix for a multi-class classification scenario

    Args:
        plot_path (str): path to which the plot will be saved
        class_names: names of the predicted classes
        y_test (np.ndarray): true labels
        y_pred (np.ndarray): predicted labels
    """
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    _, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(plot_path)


if __name__ == "__main__":
    main()
    