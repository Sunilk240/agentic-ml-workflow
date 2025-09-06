import pandas as pd
from typing import Dict, Any
from langchain_core.tools import tool
from helpers import DATAFRAME_CACHE, get_dataset, PIPELINE_STATE

@tool
def explore_dataset(path: str) -> Dict[str, Any]:
    """
    Provides comprehensive dataset overview.
    User says 'explore data' or 'show dataset info'.
    """
    df = get_dataset(path)
    if df is None:
        return {"error": "Error loading dataset"}

    # Agent response (concise)
    response = "📋 **Dataset Overview**\n\n"
    response += f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n"
    response += f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"

    response += "**📊 COLUMN INFORMATION:**\n"
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    response += f"• **Numeric Columns ({len(numeric_cols)}):** {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}\n"
    response += f"• **Categorical Columns ({len(categorical_cols)}):** {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}\n\n"

    # Missing values summary
    missing_count = df.isnull().sum().sum()
    response += f"**🔍 DATA QUALITY:**\n"
    response += f"• **Missing Values:** {missing_count} ({missing_count/len(df)*100:.1f}% of total)\n"

    duplicate_count = df.duplicated().sum()
    response += f"• **Duplicate Rows:** {duplicate_count} ({duplicate_count/len(df)*100:.1f}%)\n\n"

    response += "**💡 SUGGESTED NEXT STEPS:**\n"
    if missing_count > 0 or duplicate_count > 0:
        response += "• 'perform cleaning' - Clean data quality issues\n"
    else:
        response += "• 'perform cleaning' - Check for outliers and data quality\n"
    if categorical_cols:
        response += "• 'perform feature engineering' - Prepare features for ML\n"
    else:
        response += "• 'perform feature engineering' - Analyze feature correlations\n"
    response += "• 'train model' - Detect problem type and show available algorithms\n"

    # Collect detailed data for frontend
    detailed_data = {
        "data_preview": df.head().to_dict('records'),  # First 5 rows as list of dicts
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns_info": {
            "numeric": {
                "count": len(numeric_cols),
                "names": numeric_cols
            },
            "categorical": {
                "count": len(categorical_cols), 
                "names": categorical_cols
            }
        },
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "data_quality": {
            "missing_values": {
                "total": int(missing_count),
                "percentage": round(missing_count/len(df)*100, 1),
                "by_column": {k: int(v) for k, v in df.isnull().sum().to_dict().items()}
            },
            "duplicates": {
                "count": int(duplicate_count),
                "percentage": round(duplicate_count/len(df)*100, 1)
            }
        },
        "memory_usage": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "basic_stats": df.describe().to_dict() if len(numeric_cols) > 0 else {}
    }

    # Store detailed data for frontend (will be picked up by API)
    PIPELINE_STATE["last_detailed_data"] = detailed_data
    
    return {
        "exploration_complete": True,
        "dataset_shape": df.shape,
        "formatted_response": response,
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.to_dict()
    }
