# Basics
import numpy as np
try:
    import config
except:
    from . import config
logger = config.CTxAILogger("INFO")

# Models and data processing
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    RidgeClassifier,
    LogisticRegression,
)
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# Utils
import dask.array as da
import optuna
from optuna.samplers import TPESampler, RandomSampler
from functools import partial
from datetime import datetime
from typing import Any
from cupyx.scipy.spatial.distance import cdist

# Warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def find_best_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    """ Create an Optuna study and find the best hyperparameters

    Args:
        X_train / X_val (np.ndarray): input variables
        y_train / y_val (np.ndarray): output target values

    Returns:
        Scikit-learn model: trained model with best hyper-parameters
    """
    # Initialize objective and sampler
    cfg = config.get_config()
    match cfg["PREDICTOR_TASK_TYPE"]:
        case "classification": scoring = "f1_macro"
        case "regression": scoring = "neg_mean_squared_error" 
    objective = partial(
        optuna_objective_fn,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        scoring=scoring,
    )
    if cfg["OPTUNA_SAMPLER"] == "tpe":
        sampler = TPESampler(seed=cfg["RANDOM_STATE"])
    else:
        sampler = RandomSampler(seed=cfg["RANDOM_STATE"])
    
    # Optimizer hyper-parameters
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(
        func=objective,
        n_trials=cfg["N_PREDICTION_OPTUNA_TRIALS"],
        show_progress_bar=True,
    )
    
    return study.best_params


def optuna_objective_fn(
    trial: optuna.trial.BaseTrial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scoring: str,
) -> float:
    """ Objective function to be optimized by optuna

    Args:
        trial (optuna.trial): current optuna trial
        X_train/X_val (np.ndarray): input variables
        y_train/y_val (np.ndarray): output target values
        scoring (str): type of score used to compute the objective
        
    Returns:
        float: value for the metric to optimize
    """
    model = suggest_model(trial)
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_val, y_val, cv=5, scoring=scoring)
    
    return np.mean(scores)


def suggest_model(
    trial: optuna.trial.BaseTrial|None=None,
    model_params: dict|None=None,
):
    """ Suggest model hyper-parameters for one given trial

    Args:
        trial (optuna.trial.BaseTrial): current optuna trial
        model_params (dict): if not None, provide a model with given parameters

    Returns:
        scikit-learn model: model with suggested or provided hyper-parameters
    """
    cfg = config.get_config()
    model_name = "%s_%s" % (cfg["PREDICTOR_MODEL_TYPE"], cfg["PREDICTOR_TASK_TYPE"])
    assert not (trial is None and model_params is None)
    match model_name:
        
        #####################
        # REGRESSION MODELS #
        #####################
        
        case "ridge_regression":
            model_class = Ridge
            if model_params is None:
                model_params = {
                    "alpha": trial.suggest_float("alpha", 0.01, 10.0),
                    "solver": trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]),
                }

        case "lasso_regression":
            model_class = Lasso
            if model_params is None:
                model_params = {
                    "alpha": trial.suggest_float("alpha", 0.01, 10.0),
                    "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
                }

        case "elasticnet_regression":
            model_class = ElasticNet
            if model_params is None:
                model_params = {
                    "alpha": trial.suggest_float("alpha", 0.01, 10.0),
                    "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                    "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
                }
                    
        case "sv_regression":
            model_class = SVR
            if model_params is None:
                model_params = {
                    "C": trial.suggest_float("C", 0.1, 10.0),
                    "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
                }
        
        case "decisiontree_regression":
            model_class = DecisionTreeRegressor
            if model_params is None:
                model_params = {
                    "max_depth": trial.suggest_int("max_depth", 1, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                }
               
        case "randomforest_regression":
            model_class = RandomForestRegressor
            if model_params is None:
                model_params = {
                    "n_estimators": trial.suggest_int("n_estimators", 10, 200),
                    "max_depth": trial.suggest_int("max_depth", 1, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                }
        
        case "knn_regression":
            model_class = KNeighborsRegressor
            if model_params is None:
                model_params = {
                    "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
                    "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                    "p": trial.suggest_int("p", 1, 2),
                }
        
        #########################
        # CLASSIFICATION MODELS #
        #########################
        
        case "ridge_classification":
            model_class = RidgeClassifier
            if model_params is None:
                model_params = {
                    "alpha": trial.suggest_float("alpha", 0.01, 10.0),
                }
                
        case "lasso_classification":
            model_class = LogisticRegression
            if model_params is None:
                model_params = {
                    "penalty": trial.suggest_categorical("penalty", ["l1"]),
                    "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
                    "C": trial.suggest_float("C", 0.01, 10.0),
                }
                
        case "elasticnet_classification":
            model_class = LogisticRegression
            if model_params is None:
                model_params = {
                    "penalty": trial.suggest_categorical("penalty", ["elasticnet"]),
                    "solver": trial.suggest_categorical("solver", ["saga"]),
                    "C": trial.suggest_float("C", 0.01, 10.0),
                    "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                }

        case "sv_classification":
            model_class = SVC
            if model_params is None:
                model_params = {
                    "C": trial.suggest_float("C", 0.01, 10.0),
                    "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
                }
                if cfg["PREDICTOR_TASK_TYPE"] == "multi-class":
                    model_params["decision_function_shape"] =\
                        trial.suggest_categorical("decision_function_shape", ["ovo", "ovr"])

        case "decisiontree_classification":
            model_class = DecisionTreeClassifier
            if model_params is None:
                model_params = {
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                }

        case "randomforest_classification":
            model_class = RandomForestClassifier
            if model_params is None:
                model_params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                }

        case "gradientboosting_classification":
            model_class = GradientBoostingClassifier
            if model_params is None:
                model_params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                }

        case "knn_classification":
            model_class = KNeighborsClassifier
            if model_params is None:
                model_params = {
                    "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
                    "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                }

        case "naivebayes_classification":
            model_class = GaussianNB
            if model_params is None:
                model_params = {}

        case _:
            raise ValueError(f"Unsupported model combination: {model_name.split('_')}")

    return model_class(**model_params)
    

class UnravelTransformer(BaseEstimator, TransformerMixin):
    """ Custom transformer for reshaping
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.reshape(-1, 1)
    
    def inverse_transform(self, X):
        return X.reshape(-1)


class RavelTransformer(BaseEstimator, TransformerMixin):
    """ Custom transformer for raveling
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.ravel()
    
    def inverse_transform(self, X):
        return X.reshape(-1, 1)


class LogScalingTransformer(BaseEstimator, TransformerMixin):
    """ Custom transformer for log scaling
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(1 + X)
    
    def inverse_transform(self, X):
        return np.expm1(X) - 1


def get_ct_target(ct_data: dict) -> Any:
    """ Extract target defined by target_type from clinical trial data

    Args:
        ct_data (str): clinical trial data extracted from ct.gov json file

    Returns:
        Any: extracted target
    """
    cfg = config.get_config()
    protocol = ct_data\
        .get("FullStudy", {})\
        .get("Study", {})\
        .get("ProtocolSection", {})
    match cfg["PREDICTOR_TARGET_TYPE"]:
        
        case "phase":
            phases = protocol\
                .get("DesignModule", {})\
                .get("PhaseList", {})\
                .get("Phase", [])
            for p in phases:
                if p in ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]:
                    return p  # return first matching phase only
            return None  # if no matching phase found
        
        case "status":
            return protocol\
                .get("StatusModule", {})\
                .get("OverallStatus", "")
        
        case "condition":
            return protocol\
                .get("ConditionsModule", {})\
                .get("ConditionList", {})\
                .get("Condition", [])
        
        case "intervention_type":
            interventions = protocol\
                .get("ArmsInterventionsModule", {})\
                .get("InterventionList", {})\
                .get("Intervention", [])
            return [
                intervention.get("InterventionType", "")
                for intervention in interventions
            ]
        
        case "intervention_name":
            interventions = protocol\
                .get("ArmsInterventionsModule", {})\
                .get("InterventionList", {})\
                .get("Intervention", [])
            return [
                intervention.get("InterventionName", "")
                for intervention in interventions
            ]
        
        case "primary_outcome":
            primary_outcomes = protocol\
                .get("OutcomesModule", {})\
                .get("PrimaryOutcomeList", {})\
                .get("PrimaryOutcome", [])
            return [
                outcome.get("PrimaryOutcomeMeasure", "")
                for outcome in primary_outcomes
            ]
        
        case "is_fda_regulated_drug":
            target = protocol\
                .get("OversightModule", {})\
                .get("IsFDARegulatedDrug", "")
            if target == "": target = None
            return target
        
        case "enrollment_count":
            return protocol\
                .get("DesignModule", {})\
                .get("EnrollmentInfo", {})\
                .get("EnrollmentCount")
        
        case "enrollment_type":
            return protocol\
                .get("DesignModule", {})\
                .get("EnrollmentInfo", {})\
                .get("EnrollmentType")
        
        case "enrollment_rate":
            
            def parse_date(date_str: str):
                for fmt in ('%B %d, %Y', '%B %Y'):
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        pass
                return None
            
            enrollment_count = protocol\
                .get("DesignModule", {})\
                .get("EnrollmentInfo", {})\
                .get("EnrollmentCount")
                    
            start_date_str = protocol\
                .get("StatusModule", {})\
                .get("StartDateStruct", {})\
                .get("StartDate")
            
            completion_date_str = protocol\
                .get("StatusModule", {})\
                .get("CompletionDateStruct", {})\
                .get("CompletionDate")
            
            if not all([enrollment_count, start_date_str, completion_date_str]):
                return None
            
            enrollment_count = int(enrollment_count)
            start_date = parse_date(start_date_str)
            completion_date = parse_date(completion_date_str)
            duration_in_months = (completion_date.year - start_date.year) * 12\
                               + (completion_date.month - start_date.month) + 1
            
            enrollment_rate = enrollment_count / duration_in_months
            return enrollment_rate
            # if enrollment_rate > 5.0:
            #     return "quick"
            # else:
            #     return "slow"
        
        case _:
            raise ValueError("Invalid label type")


def post_process_ct_targets(ct_targets: np.ndarray) -> np.ndarray:
    """ Some labels must be post-processed, for example enrollment rate that need
        to know the whole distribution to apply the pareto principle

    Args:
        ct_targets (np.ndarray): target array to post-process (or not)

    Returns:
        np.ndarray: post-processed target array
    """
    cfg = config.get_config()
    
    if cfg["PREDICTOR_TARGET_TYPE"] == "enrollment_rate":
        threshold = np.percentile(ct_targets, 50)  # 80
        return (ct_targets > threshold).astype(int)
    
    return ct_targets
    

def select_normalizers():
    """ Select  input features and output targets (depending on label type)
    """
    cfg = config.get_config()
    normalizer_X = StandardScaler()
    
    # Normalize output targets
    match cfg["PREDICTOR_TARGET_TYPE"]:
        
        case "debug":
            identity = lambda x: x
            normalizer_y = FunctionTransformer(func=identity, inverse_func=identity)
        
        case "enrollment_count":
            normalizer_y = Pipeline([
                ('reshape', UnravelTransformer()),
                ("log", LogScalingTransformer()),
                ("scaler", StandardScaler()),
                ('ravel', RavelTransformer())
            ])
        
        case "phase" | "status" | "study_type" | "condition" | "intervention_type"\
            | "intervention_name" | "primary_outcome" | "is_fda_regulated_drug"\
            | "enrollment_count" |  "enrollment_type" | "enrollment_rate":
            normalizer_y = LabelEncoder()
        
        case _:
            raise ValueError("Invalid label type")
    
    return normalizer_X, normalizer_y


def compute_medoid(data: np.ndarray) -> np.ndarray:
    """ Compute medoids of a subset of samples of shape (n_samples, n_features)
        Distance computations are made with dask to mitigate memory requirements
    """
    dask_data = da.from_array(data, chunks=1_000)
    def compute_distances(chunk): return cdist(chunk, data)
    distances = dask_data.map_blocks(compute_distances, dtype=float)
    sum_distances = distances.sum(axis=1).compute()
    medoid_index = sum_distances.argmin().get()
    return data[medoid_index]