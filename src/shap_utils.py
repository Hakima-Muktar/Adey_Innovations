"""
SHAP utilities for model explainability.

This module provides helper functions for generating SHAP explanations
from sklearn/imblearn pipelines, including feature name extraction,
value computation, and various plot types.
"""

from typing import List, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def _shap_values_to_2d(
    shap_values: shap.Explanation,
    positive_class_index: int = 1,
) -> np.ndarray:
    """Normalize SHAP values to a 2D array of shape (n_samples, n_features).

    SHAP can return:
    - (n_samples, n_features)
    - (n_samples, n_classes, n_features)
    - (n_samples, n_features, n_classes)

    For binary classification we prefer the SHAP values for the positive class.
    """
    values = np.asarray(shap_values.values)

    if values.ndim == 2:
        return values

    if values.ndim != 3:
        raise ValueError(f"Unexpected SHAP values shape: {values.shape}")

    # Case A: (n_samples, n_classes, n_features)
    if values.shape[2] != 0 and values.shape[2] != 1 and values.shape[2] == np.asarray(shap_values.data).shape[1]:
        class_idx = min(positive_class_index, values.shape[1] - 1)
        return values[:, class_idx, :]

    # Case B: (n_samples, n_features, n_classes)
    if values.shape[1] == np.asarray(shap_values.data).shape[1]:
        class_idx = min(positive_class_index, values.shape[2] - 1)
        return values[:, :, class_idx]

    # Fallback: try to infer by matching feature dimension
    if values.shape[2] == np.asarray(shap_values.data).shape[1]:
        class_idx = min(positive_class_index, values.shape[1] - 1)
        return values[:, class_idx, :]
    if values.shape[1] == np.asarray(shap_values.data).shape[1]:
        class_idx = min(positive_class_index, values.shape[2] - 1)
        return values[:, :, class_idx]

    raise ValueError(
        f"Cannot normalize SHAP values of shape {values.shape} to 2D with data shape {np.asarray(shap_values.data).shape}."
    )


def _shap_base_values_to_1d(
    shap_values: shap.Explanation,
    positive_class_index: int = 1,
) -> np.ndarray:
    """Normalize SHAP base_values to 1D of length n_samples (positive class if needed)."""
    base = np.asarray(shap_values.base_values)

    if base.ndim == 0:
        return np.repeat(base, repeats=np.asarray(shap_values.data).shape[0])

    if base.ndim == 1:
        return base

    if base.ndim == 2:
        # (n_samples, n_classes)
        class_idx = min(positive_class_index, base.shape[1] - 1)
        return base[:, class_idx]

    raise ValueError(f"Unexpected SHAP base_values shape: {base.shape}")


def get_feature_names_from_pipeline(
    pipeline: Any,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[str]:
    """
    Extract feature names after preprocessing (including one-hot encoding).

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn/imblearn pipeline with a 'preprocessor' step.
    numeric_features : List[str]
        Original numeric feature names.
    categorical_features : List[str]
        Original categorical feature names.

    Returns
    -------
    List[str]
        Full list of feature names after transformation.
    """
    if "preprocessor" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["preprocessor"]
        cat_encoder = preprocessor.named_transformers_.get("cat")
        if cat_encoder is not None:
            cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
        else:
            cat_feature_names = []
        return numeric_features + cat_feature_names
    else:
        # No preprocessor (e.g., creditcard pipeline with just scaler)
        return numeric_features + categorical_features


def transform_for_explanation(
    pipeline: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Transform features using only the preprocessing steps (not SMOTE or classifier).

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline.
    X : pd.DataFrame
        Raw features to transform.

    Returns
    -------
    np.ndarray
        Transformed feature array ready for SHAP.
    """
    if "preprocessor" in pipeline.named_steps:
        return pipeline.named_steps["preprocessor"].transform(X)
    elif "scaler" in pipeline.named_steps:
        return pipeline.named_steps["scaler"].transform(X)
    else:
        return X.values


def create_tree_explainer(
    pipeline: Any,
    X_background: Optional[np.ndarray] = None,
) -> shap.Explainer:
    """
    Create a SHAP TreeExplainer for tree-based models in a pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline with a 'classifier' step.
    X_background : np.ndarray, optional
        Background dataset for explainer. If None, uses model internals.

    Returns
    -------
    shap.Explainer
        SHAP explainer for the classifier.
    """
    classifier = pipeline.named_steps["classifier"]

    # TreeExplainer works for RandomForest, GradientBoosting, XGBoost, LightGBM
    if X_background is not None:
        return shap.TreeExplainer(classifier, X_background)
    else:
        return shap.TreeExplainer(classifier)


def create_linear_explainer(
    pipeline: Any,
    X_background: np.ndarray,
) -> shap.Explainer:
    """
    Create a SHAP LinearExplainer for linear models in a pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline with a 'classifier' step.
    X_background : np.ndarray
        Background dataset for explainer.

    Returns
    -------
    shap.Explainer
        SHAP explainer for the linear classifier.
    """
    classifier = pipeline.named_steps["classifier"]
    return shap.LinearExplainer(classifier, X_background)


def compute_shap_values(
    explainer: shap.Explainer,
    X_transformed: np.ndarray,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> shap.Explanation:
    """
    Compute SHAP values for transformed features.

    Parameters
    ----------
    explainer : shap.Explainer
        SHAP explainer.
    X_transformed : np.ndarray
        Transformed feature array.
    sample_size : int, optional
        If provided, sample this many rows for faster computation.
    random_state : int
        Random state for sampling.

    Returns
    -------
    shap.Explanation
        SHAP explanation object.
    """
    if sample_size is not None and sample_size < len(X_transformed):
        np.random.seed(random_state)
        indices = np.random.choice(len(X_transformed), size=sample_size, replace=False)
        X_sample = X_transformed[indices]
    else:
        X_sample = X_transformed

    return explainer(X_sample)


def plot_shap_summary(
    shap_values: shap.Explanation,
    feature_names: List[str],
    max_display: int = 20,
    plot_type: str = "dot",
    title: str = "SHAP Summary Plot",
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate SHAP summary plot (beeswarm or bar).

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP explanation object.
    feature_names : List[str]
        Feature names.
    max_display : int
        Maximum features to display.
    plot_type : str
        'dot' for beeswarm, 'bar' for bar plot.
    title : str
        Plot title.
    save_path : Path, optional
        If provided, save figure to this path.
    """
    values_2d = _shap_values_to_2d(shap_values)
    data_2d = np.asarray(shap_values.data)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        values_2d,
        features=data_2d,
        feature_names=feature_names,
        max_display=max_display,
        plot_type=plot_type,
        show=False,
    )
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_shap_bar(
    shap_values: shap.Explanation,
    feature_names: List[str],
    max_display: int = 20,
    title: str = "SHAP Feature Importance",
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate SHAP bar plot showing mean absolute SHAP values.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP explanation object.
    feature_names : List[str]
        Feature names.
    max_display : int
        Maximum features to display.
    title : str
        Plot title.
    save_path : Path, optional
        If provided, save figure to this path.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature importance values.
    """
    values_2d = _shap_values_to_2d(shap_values)

    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(values_2d).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    # Plot
    top_features = importance_df.head(max_display)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_features["feature"], top_features["mean_abs_shap"], color="#e74c3c")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()

    return importance_df


def plot_shap_dependence(
    shap_values: shap.Explanation,
    feature_name: str,
    feature_names: List[str],
    interaction_feature: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate SHAP dependence plot for a specific feature.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP explanation object.
    feature_name : str
        Feature to plot.
    feature_names : List[str]
        All feature names.
    interaction_feature : str, optional
        Feature for interaction coloring.
    title : str, optional
        Plot title.
    save_path : Path, optional
        If provided, save figure to this path.
    """
    if feature_name not in feature_names:
        print(f"Feature '{feature_name}' not found in feature list.")
        return

    feature_idx = feature_names.index(feature_name)

    plt.figure(figsize=(10, 6))

    values_2d = _shap_values_to_2d(shap_values)
    data_2d = np.asarray(shap_values.data)

    if interaction_feature and interaction_feature in feature_names:
        interaction_idx = feature_names.index(interaction_feature)
        shap.dependence_plot(
            feature_idx,
            values_2d,
            data_2d,
            feature_names=feature_names,
            interaction_index=interaction_idx,
            show=False,
        )
    else:
        shap.dependence_plot(
            feature_idx,
            values_2d,
            data_2d,
            feature_names=feature_names,
            show=False,
        )

    if title:
        plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_shap_waterfall(
    shap_values: shap.Explanation,
    index: int,
    feature_names: List[str],
    max_display: int = 15,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate SHAP waterfall plot for a single prediction.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP explanation object.
    index : int
        Index of the sample to explain.
    feature_names : List[str]
        Feature names.
    max_display : int
        Maximum features to display.
    title : str, optional
        Plot title.
    save_path : Path, optional
        If provided, save figure to this path.
    """
    values_2d = _shap_values_to_2d(shap_values)
    base_1d = _shap_base_values_to_1d(shap_values)
    data_2d = np.asarray(shap_values.data)

    # Create single-row explanation
    single_exp = shap.Explanation(
        values=values_2d[index],
        base_values=base_1d[index],
        data=data_2d[index],
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(single_exp, max_display=max_display, show=False)

    if title:
        plt.title(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def get_example_cases(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_examples: int = 3,
) -> Dict[str, List[int]]:
    """
    Get indices of example cases for local explanations.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray
        Predicted probabilities for positive class.
    n_examples : int
        Number of examples per category.

    Returns
    -------
    Dict[str, List[int]]
        Dictionary with keys 'true_positive', 'false_positive', 'false_negative', 'true_negative'
        and indices as values.
    """
    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tn_mask = (y_true == 0) & (y_pred == 0)

    def get_top_indices(mask, proba, n, ascending=False):
        """Get indices sorted by probability."""
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []
        proba_subset = proba[indices]
        if ascending:
            sorted_idx = np.argsort(proba_subset)[:n]
        else:
            sorted_idx = np.argsort(proba_subset)[-n:][::-1]
        return indices[sorted_idx].tolist()

    return {
        "true_positive": get_top_indices(tp_mask, y_proba, n_examples, ascending=False),
        "false_positive": get_top_indices(fp_mask, y_proba, n_examples, ascending=False),
        "false_negative": get_top_indices(fn_mask, y_proba, n_examples, ascending=True),
        "true_negative": get_top_indices(tn_mask, y_proba, n_examples, ascending=True),
    }


def explain_single_prediction(
    pipeline: Any,
    X_single: pd.DataFrame,
    feature_names: List[str],
    explainer: shap.Explainer,
) -> Dict[str, Any]:
    """
    Generate explanation for a single prediction.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline.
    X_single : pd.DataFrame
        Single row of features.
    feature_names : List[str]
        Feature names after transformation.
    explainer : shap.Explainer
        SHAP explainer.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing prediction info and SHAP values.
    """
    X_trans = transform_for_explanation(pipeline, X_single)
    shap_vals = explainer(X_trans)

    proba = pipeline.predict_proba(X_single)[0, 1]
    pred = int(proba >= 0.5)

    return {
        "prediction": pred,
        "probability": proba,
        "shap_values": shap_vals,
        "feature_names": feature_names,
    }