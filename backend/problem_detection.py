import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from helpers import DATAFRAME_CACHE, PIPELINE_STATE

def _analyze_target_column(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Analyze target column to determine problem type"""
    analysis = {}
    
    target_series = df[target_col]
    
    # Basic statistics
    analysis['column_name'] = target_col
    analysis['data_type'] = str(target_series.dtype)
    analysis['unique_values'] = target_series.nunique()
    analysis['total_values'] = len(target_series)
    analysis['unique_ratio'] = analysis['unique_values'] / analysis['total_values']
    analysis['null_count'] = target_series.isnull().sum()
    
    # Sample values
    analysis['sample_values'] = target_series.dropna().unique()[:10].tolist()
    
    # Value distribution
    if analysis['unique_values'] <= 20:
        analysis['value_counts'] = target_series.value_counts().head(10).to_dict()
    
    return analysis

def _determine_problem_type(target_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Determine problem type based on target analysis"""
    
    data_type = target_analysis['data_type']
    unique_ratio = target_analysis['unique_ratio']
    unique_values = target_analysis['unique_values']
    sample_values = target_analysis['sample_values']
    
    # Classification indicators
    classification_score = 0
    regression_score = 0
    
    # Data type analysis
    if 'object' in data_type or 'category' in data_type:
        classification_score += 3
    elif 'int' in data_type:
        if unique_values <= 10:
            classification_score += 2
        else:
            regression_score += 1
    elif 'float' in data_type:
        regression_score += 2
    
    # Unique ratio analysis
    if unique_ratio < 0.05:  # Less than 5% unique values
        classification_score += 3
    elif unique_ratio < 0.1:  # Less than 10% unique values
        classification_score += 2
    elif unique_ratio > 0.5:  # More than 50% unique values
        regression_score += 2
    
    # Unique count analysis
    if unique_values <= 2:
        classification_score += 3
        problem_subtype = "binary_classification"
    elif unique_values <= 10:
        classification_score += 2
        problem_subtype = "multiclass_classification"
    elif unique_values > 50:
        regression_score += 2
        problem_subtype = "continuous_regression"
    
    # Sample values analysis
    if len(sample_values) > 0:
        # Check if values look like categories
        categorical_indicators = ['yes', 'no', 'true', 'false', 'male', 'female', 
                                'high', 'medium', 'low', 'good', 'bad', 'positive', 'negative']
        
        sample_str = [str(v).lower() for v in sample_values[:5]]
        if any(indicator in ' '.join(sample_str) for indicator in categorical_indicators):
            classification_score += 2
    
    # Final determination
    if classification_score > regression_score:
        if unique_values <= 2:
            return {
                "problem_type": "classification",
                "subtype": "binary_classification",
                "confidence": min(classification_score / (classification_score + regression_score + 1), 0.95),
                "reasoning": f"Target has {unique_values} unique values with {unique_ratio:.1%} uniqueness ratio"
            }
        else:
            return {
                "problem_type": "classification", 
                "subtype": "multiclass_classification",
                "confidence": min(classification_score / (classification_score + regression_score + 1), 0.95),
                "reasoning": f"Target has {unique_values} unique values with {unique_ratio:.1%} uniqueness ratio"
            }
    else:
        return {
            "problem_type": "regression",
            "subtype": "continuous_regression",
            "confidence": min(regression_score / (classification_score + regression_score + 1), 0.95),
            "reasoning": f"Target has {unique_values} unique values with {unique_ratio:.1%} uniqueness ratio"
        }

def _suggest_algorithms(problem_type: str, subtype: str, dataset_size: int, feature_count: int) -> Dict[str, List[str]]:
    """Suggest appropriate algorithms based on problem type and dataset characteristics"""
    
    algorithms = {
        "recommended": [],
        "alternative": [],
        "advanced": []
    }
    
    if problem_type == "classification":
        if subtype == "binary_classification":
            algorithms["recommended"] = [
                "logistic_regression",
                "random_forest", 
                "gradient_boosting"
            ]
            algorithms["alternative"] = [
                "svm",
                "knn",
                "naive_bayes"
            ]
            algorithms["advanced"] = [
                "xgboost",
                "lightgbm",
                "adaboost"
            ]
        else:  # multiclass
            algorithms["recommended"] = [
                "random_forest",
                "gradient_boosting",
                "logistic_regression"
            ]
            algorithms["alternative"] = [
                "svm",
                "knn",
                "decision_tree"
            ]
            algorithms["advanced"] = [
                "xgboost",
                "lightgbm",
                "extra_trees"
            ]
    
    elif problem_type == "regression":
        algorithms["recommended"] = [
            "random_forest",
            "gradient_boosting",
            "linear_regression"
        ]
        algorithms["alternative"] = [
            "ridge_regression",
            "lasso_regression",
            "svr"
        ]
        algorithms["advanced"] = [
            "xgboost",
            "lightgbm",
            "elastic_net"
        ]
    
    elif problem_type == "clustering":
        algorithms["recommended"] = [
            "kmeans",
            "hierarchical",
            "dbscan"
        ]
        algorithms["alternative"] = [
            "gaussian_mixture",
            "spectral_clustering",
            "mean_shift"
        ]
        algorithms["advanced"] = [
            "hdbscan",
            "optics",
            "birch"
        ]
    
    # Adjust based on dataset characteristics
    if dataset_size < 1000:
        # For small datasets, simpler models often work better
        if "knn" in algorithms["alternative"]:
            algorithms["recommended"].append("knn")
    
    if feature_count > 100:
        # For high-dimensional data, regularized models work better
        if problem_type == "regression":
            algorithms["recommended"] = ["ridge_regression", "lasso_regression"] + algorithms["recommended"]
    
    return algorithms

@tool
def detect_problem_type(path: str, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes dataset to determine if it's a classification, regression, or clustering problem.
    User says 'detect problem type' or 'what type of problem is this'.
    """
    if path not in DATAFRAME_CACHE:
        return {"error": "Dataset not found. Please load dataset first."}

    df = DATAFRAME_CACHE[path]
    
    # Auto-detect target column if not specified
    if not target_column:
        # Look for common target column names
        potential_targets = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['target', 'label', 'y', 'class', 'category', 
                                                       'price', 'value', 'score', 'rating', 'outcome']):
                potential_targets.append(col)
        
        if potential_targets:
            target_column = potential_targets[0]
        else:
            # Use last column as default
            target_column = df.columns[-1]
    
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in dataset"}
    
    # Analyze target column
    target_analysis = _analyze_target_column(df, target_column)
    
    # Determine problem type
    problem_info = _determine_problem_type(target_analysis)
    
    # Get dataset characteristics
    dataset_size = len(df)
    feature_count = len(df.columns) - 1  # Excluding target
    
    # Suggest algorithms
    suggested_algorithms = _suggest_algorithms(
        problem_info["problem_type"], 
        problem_info["subtype"],
        dataset_size,
        feature_count
    )
    
    # Check if clustering might be appropriate (no clear target)
    clustering_indicators = 0
    if target_analysis['unique_ratio'] > 0.8:  # Very high uniqueness
        clustering_indicators += 1
    if 'id' in target_column.lower() or 'index' in target_column.lower():
        clustering_indicators += 2
    
    clustering_suggestion = clustering_indicators >= 2
    
    # Store analysis
    PIPELINE_STATE["problem_analysis"] = {
        "target_column": target_column,
        "problem_type": problem_info["problem_type"],
        "subtype": problem_info["subtype"],
        "confidence": problem_info["confidence"],
        "target_analysis": target_analysis,
        "suggested_algorithms": suggested_algorithms,
        "clustering_possible": clustering_suggestion
    }
    
    # Format response
    response = "🔍 **Problem Type Detection Complete**\n\n"
    response += f"**Target Column:** {target_column}\n"
    response += f"**Problem Type:** {problem_info['problem_type'].title()}\n"
    response += f"**Subtype:** {problem_info['subtype'].replace('_', ' ').title()}\n"
    response += f"**Confidence:** {problem_info['confidence']:.1%}\n\n"
    
    response += "**📊 TARGET ANALYSIS:**\n"
    response += f"• **Data Type:** {target_analysis['data_type']}\n"
    response += f"• **Unique Values:** {target_analysis['unique_values']:,} ({target_analysis['unique_ratio']:.1%} of total)\n"
    response += f"• **Sample Values:** {', '.join(map(str, target_analysis['sample_values'][:5]))}\n"
    
    if 'value_counts' in target_analysis:
        response += f"• **Distribution:** {dict(list(target_analysis['value_counts'].items())[:3])}\n"
    
    response += f"\n**🎯 REASONING:** {problem_info['reasoning']}\n\n"
    
    response += "**🤖 RECOMMENDED ALGORITHMS:**\n"
    for alg in suggested_algorithms["recommended"]:
        response += f"• {alg.replace('_', ' ').title()}\n"
    
    response += "\n**🔄 ALTERNATIVE OPTIONS:**\n"
    for alg in suggested_algorithms["alternative"]:
        response += f"• {alg.replace('_', ' ').title()}\n"
    
    if clustering_suggestion:
        response += "\n**💡 CLUSTERING OPTION:**\n"
        response += "• Dataset might be suitable for clustering analysis\n"
        response += "• Consider unsupervised learning approaches\n"
    
    response += "\n**💡 NEXT STEPS:**\n"
    response += "• Type 'train [algorithm_name]' to train specific model\n"
    response += "• Type 'train recommended' to train all recommended models\n"
    response += "• Type 'show algorithms' to see all available options"
    
    return {
        "detection_complete": True,
        "problem_type": problem_info["problem_type"],
        "subtype": problem_info["subtype"],
        "confidence": problem_info["confidence"],
        "target_column": target_column,
        "formatted_response": response,
        "suggested_algorithms": suggested_algorithms
    }

@tool
def show_available_algorithms(problem_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Shows all available algorithms for the detected or specified problem type.
    User says 'show algorithms' or 'what models are available'.
    """
    if not problem_type and "problem_analysis" in PIPELINE_STATE:
        problem_type = PIPELINE_STATE["problem_analysis"]["problem_type"]
    
    if not problem_type:
        return {"error": "Please detect problem type first using 'detect problem type'"}
    
    # Get all available algorithms
    all_algorithms = {
        "classification": {
            "Basic Models": [
                "logistic_regression - Fast, interpretable, good baseline",
                "decision_tree - Interpretable, handles non-linear patterns",
                "naive_bayes - Fast, works well with small datasets"
            ],
            "Ensemble Models": [
                "random_forest - Robust, handles overfitting well",
                "gradient_boosting - High performance, sequential learning",
                "extra_trees - Fast random forest variant"
            ],
            "Distance-Based": [
                "knn - Simple, non-parametric, good for local patterns",
                "svm - Effective for high-dimensional data"
            ],
            "Advanced": [
                "xgboost - State-of-the-art gradient boosting",
                "lightgbm - Fast gradient boosting",
                "adaboost - Adaptive boosting ensemble"
            ]
        },
        "regression": {
            "Linear Models": [
                "linear_regression - Simple, interpretable baseline",
                "ridge_regression - Regularized, prevents overfitting",
                "lasso_regression - Feature selection + regularization",
                "elastic_net - Combines Ridge and Lasso"
            ],
            "Ensemble Models": [
                "random_forest - Robust, handles non-linearity",
                "gradient_boosting - High performance sequential learning",
                "extra_trees - Fast ensemble method"
            ],
            "Distance-Based": [
                "knn - Non-parametric, local patterns",
                "svr - Support Vector Regression"
            ],
            "Advanced": [
                "xgboost - State-of-the-art gradient boosting",
                "lightgbm - Fast and efficient",
                "catboost - Handles categorical features well"
            ]
        },
        "clustering": {
            "Centroid-Based": [
                "kmeans - Fast, works well with spherical clusters",
                "kmeans_plus - Improved initialization",
                "mini_batch_kmeans - Scalable for large datasets"
            ],
            "Hierarchical": [
                "hierarchical - Creates cluster dendrograms",
                "birch - Memory efficient hierarchical"
            ],
            "Density-Based": [
                "dbscan - Finds arbitrary shaped clusters",
                "hdbscan - Hierarchical density-based",
                "optics - Ordering points for cluster structure"
            ],
            "Distribution-Based": [
                "gaussian_mixture - Probabilistic clustering",
                "spectral_clustering - Graph-based clustering"
            ]
        }
    }
    
    if problem_type not in all_algorithms:
        return {"error": f"Unknown problem type: {problem_type}"}
    
    algorithms = all_algorithms[problem_type]
    
    response = f"🤖 **Available {problem_type.title()} Algorithms**\n\n"
    
    for category, algs in algorithms.items():
        response += f"**{category}:**\n"
        for alg in algs:
            response += f"• {alg}\n"
        response += "\n"
    
    response += "**💡 USAGE:**\n"
    response += "• Type 'train [algorithm_name]' for specific model\n"
    response += "• Type 'train recommended' for best performers\n"
    response += "• Type 'compare algorithms' to train multiple models\n"
    
    return {
        "algorithms_shown": True,
        "problem_type": problem_type,
        "formatted_response": response,
        "available_algorithms": algorithms
    }