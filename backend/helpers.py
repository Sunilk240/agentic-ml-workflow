import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# =====================================================
# GLOBAL STATE MANAGEMENT
# =====================================================

DATAFRAME_CACHE = {}
PIPELINE_STATE = {
    "current_stage": None,
    "completed_operations": [],
    "dataset_info": {},
    "cleaning_analysis": {},
    "feature_analysis": {},
    "models_trained": {},
    "evaluation_results": {}
}

# =====================================================
# INTERNAL HELPER FUNCTIONS
# =====================================================

def _detect_data_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """Internal function to detect all data quality issues."""
    issues = {}

    # Missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        missing_cols = df.columns[df.isnull().any()].tolist()
        missing_percentages = (df.isnull().sum() / len(df) * 100).round(2)
        issues["missing_values"] = {
            "total": int(missing_count),
            "columns": missing_cols,
            "percentages": {col: missing_percentages[col] for col in missing_cols if missing_percentages[col] > 0}
        }

    # Duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues["duplicates"] = {
            "count": int(duplicate_count),
            "percentage": round(duplicate_count / len(df) * 100, 2)
        }

    # Outliers
    numeric_cols = df.select_dtypes(include=['number']).columns
    outlier_cols = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            outlier_cols[col] = {
                "count": len(outliers),
                "percentage": round(len(outliers) / len(df) * 100, 2)
            }

    if outlier_cols:
        issues["outliers"] = {"columns": outlier_cols}

    return issues

def _analyze_features(df: pd.DataFrame) -> Dict[str, Any]:
    """Internal function to analyze features for engineering."""
    analysis = {}

    # Feature types
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # High cardinality categoricals
    high_cardinality = []
    for col in categorical_features:
        if df[col].nunique() > len(df) * 0.5:
            high_cardinality.append(col)

    # Features needing scaling
    needs_scaling = []
    for col in numeric_features:
        if df[col].std() > 1000 or (df[col].max() - df[col].min()) > 1000:
            needs_scaling.append(col)

    # Highly correlated features
    high_corr_pairs = []
    if len(numeric_features) > 1:
        corr_matrix = df[numeric_features].corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

    analysis = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "high_cardinality": high_cardinality,
        "needs_scaling": needs_scaling,
        "high_correlations": high_corr_pairs
    }

    return analysis

def get_dataset(path: str) -> Optional[pd.DataFrame]:
    """Get dataset from cache or load from uploaded file path"""
    # First try exact path match in cache
    if path in DATAFRAME_CACHE:
        return DATAFRAME_CACHE[path]
    
    # If not found by exact path, try to find by filename
    # This handles cases where agent passes just filename but cache has full path
    import os
    filename = os.path.basename(path)
    for cached_path, df in DATAFRAME_CACHE.items():
        if os.path.basename(cached_path) == filename:
            print(f"🔧 Found dataset by filename match: {filename} -> {cached_path}")
            return df
    
    # If still not found, try to load directly from the provided path
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            DATAFRAME_CACHE[path] = df
            return df
        else:
            print(f"❌ Error loading dataset: File not found - {path}")
            print(f"🔧 Available in cache: {list(DATAFRAME_CACHE.keys())}")
            return None
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None

def get_cache_key(path: str) -> Optional[str]:
    """Get the actual cache key for a given path (handles filename vs full path)"""
    # First try exact path match
    if path in DATAFRAME_CACHE:
        return path
    
    # Try to find by filename
    import os
    filename = os.path.basename(path)
    for cached_path in DATAFRAME_CACHE.keys():
        if os.path.basename(cached_path) == filename:
            return cached_path
    
    return None

def update_pipeline_state(operation: str, data: Dict[str, Any]):
    """Update the global pipeline state"""
    PIPELINE_STATE[operation] = data
    if operation not in PIPELINE_STATE["completed_operations"]:
        PIPELINE_STATE["completed_operations"].append(operation)

def store_detailed_data_for_frontend(session_id: str, detailed_data: Dict[str, Any]):
    """Store detailed data for frontend display"""
    # Import here to avoid circular imports
    import app
    app.last_tool_detailed_data[session_id] = detailed_data
