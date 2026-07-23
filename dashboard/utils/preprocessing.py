import pandas as pd
import numpy as np

try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    pass

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Feature exclusion definitions based on notebook
CALORIES_BURNED_EXCLUDE = [
    "calories_burned",
    "required_calories",
    "BMR",
    "TDEE",
    "activity_factor",
    "goal"
]

REQUIRED_CALORIES_EXCLUDE = [
    "required_calories",
    "BMR",
    "TDEE",
    "activity_factor",
    "goal"
]

def encode_categorical_features(X_train, X_test, categorical_columns):
    """Fit one LabelEncoder per feature on training data and encode both splits."""
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    encoders = {}

    for column in categorical_columns:
        encoder = LabelEncoder()
        train_values = X_train_encoded[column].astype(str)
        test_values = X_test_encoded[column].astype(str)

        encoder.fit(train_values)
        X_train_encoded[column] = encoder.transform(train_values)

        category_map = {label: index for index, label in enumerate(encoder.classes_)}
        X_test_encoded[column] = test_values.map(category_map).fillna(-1).astype(int)
        encoders[column] = encoder

    return X_train_encoded, X_test_encoded, encoders

def scale_numerical_features(X_train, X_test, numerical_columns):
    """Standardize only numerical predictors, using statistics from training data."""
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    scaler = StandardScaler()

    X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])
    return X_train_scaled, X_test_scaled, scaler

def get_metrics(y_train, train_preds, y_test, test_preds):
    """Computes comprehensive regression metrics with alias keys for safe access."""
    tr_r2 = float(r2_score(y_train, train_preds))
    tr_mae = float(mean_absolute_error(y_train, train_preds))
    tr_mse = float(mean_squared_error(y_train, train_preds))
    tr_rmse = float(np.sqrt(tr_mse))

    te_r2 = float(r2_score(y_test, test_preds))
    te_mae = float(mean_absolute_error(y_test, test_preds))
    te_mse = float(mean_squared_error(y_test, test_preds))
    te_rmse = float(np.sqrt(te_mse))

    return {
        # Standard Test Metrics
        "test_r2": te_r2,
        "test_mae": te_mae,
        "test_mse": te_mse,
        "test_rmse": te_rmse,
        # Train Metrics
        "train_r2": tr_r2,
        "train_mae": tr_mae,
        "train_mse": tr_mse,
        "train_rmse": tr_rmse,
        # Aliases for Safety
        "r2": te_r2,
        "mae": te_mae,
        "mse": te_mse,
        "rmse": te_rmse,
        "R2": te_r2,
        "MAE": te_mae,
        "MSE": te_mse,
        "RMSE": te_rmse,
        "root_mean_squared_error": te_rmse,
        "validation_rmse": te_rmse,
        "validation_r2": te_r2,
        "validation_mae": te_mae,
        "validation_mse": te_mse
    }

def safe_get_metric(metrics_dict, metric_type="r2", default="Metric unavailable"):
    """
    Defensively retrieves a metric value checking all possible key variants.
    Prevents KeyError crashes and unhandled tracebacks.
    """
    if not isinstance(metrics_dict, dict):
        return default

    key_groups = {
        "r2": ["test_r2", "r2", "R2", "r2_score", "validation_r2", "train_r2"],
        "mae": ["test_mae", "mae", "MAE", "mean_absolute_error", "validation_mae", "train_mae"],
        "rmse": ["test_rmse", "rmse", "RMSE", "root_mean_squared_error", "validation_rmse", "train_rmse"],
        "mse": ["test_mse", "mse", "MSE", "mean_squared_error", "validation_mse", "train_mse"]
    }

    candidates = key_groups.get(str(metric_type).lower(), [str(metric_type)])
    for k in candidates:
        if k in metrics_dict and metrics_dict[k] is not None:
            val = metrics_dict[k]
            if isinstance(val, (int, float, np.number)):
                return float(val)
            return str(val)

    # Check direct dictionary key if passed explicitly
    if metric_type in metrics_dict and metrics_dict[metric_type] is not None:
        return metrics_dict[metric_type]

    return default

def safe_get_model_metrics(model_results, model_key="calories_burned"):
    """
    Defensively retrieves model evaluation metrics for model_key without raising KeyError.
    Checks 'best_metrics', '[best_model_name]["metrics"]', 'metrics', 'evaluation', etc.
    Returns (best_model_name, metrics_dict).
    """
    if not isinstance(model_results, dict) or model_key not in model_results:
        return "Unknown Model", {}

    task_results = model_results.get(model_key, {})
    if not isinstance(task_results, dict):
        return "Unknown Model", {}

    best_model_name = task_results.get("best_model_name", "Random Forest")
    
    # Candidate metric keys
    metrics = task_results.get("best_metrics")
    if metrics and isinstance(metrics, dict):
        return best_model_name, metrics

    best_model_dict = task_results.get(best_model_name, {})
    if isinstance(best_model_dict, dict) and "metrics" in best_model_dict:
        return best_model_name, best_model_dict["metrics"]

    # Fallback to any metrics / evaluation / performance key
    for alt_key in ["metrics", "evaluation", "performance", "model_metrics", "scores"]:
        if alt_key in task_results and isinstance(task_results[alt_key], dict):
            return best_model_name, task_results[alt_key]

    return best_model_name, {}
