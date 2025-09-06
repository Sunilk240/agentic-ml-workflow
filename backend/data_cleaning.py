import pandas as pd
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from helpers import DATAFRAME_CACHE, PIPELINE_STATE, _detect_data_issues

def _execute_cleaning(df: pd.DataFrame, cleaning_plan: Dict[str, Any]) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Internal function to execute cleaning operations."""
    results = {}
    original_shape = df.shape

    # Handle missing values
    if "missing_values" in cleaning_plan:
        method = cleaning_plan["missing_values"]["method"]
        columns = cleaning_plan["missing_values"].get("columns", None)

        if method == "fill_mean":
            target_cols = columns if columns else df.select_dtypes(include=['number']).columns
            for col in target_cols:
                if col in df.columns:
                    df[col].fillna(df[col].mean(), inplace=True)
        elif method == "fill_median":
            target_cols = columns if columns else df.select_dtypes(include=['number']).columns
            for col in target_cols:
                if col in df.columns:
                    df[col].fillna(df[col].median(), inplace=True)
        elif method == "drop":
            df = df.dropna(subset=columns) if columns else df.dropna()

        results["missing_values"] = f"Applied {method} method"

    # Handle duplicates
    if "duplicates" in cleaning_plan:
        method = cleaning_plan["duplicates"]["method"]
        if method == "remove_first":
            df = df.drop_duplicates(keep='first')
        elif method == "remove_last":
            df = df.drop_duplicates(keep='last')
        results["duplicates"] = f"Applied {method} method"

    # Handle outliers
    if "outliers" in cleaning_plan:
        method = cleaning_plan["outliers"]["method"]
        columns = cleaning_plan["outliers"]["columns"]

        for col in columns:
            if col not in df.columns:
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            if method == "remove":
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif method == "cap":
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == "replace_median":
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                df.loc[outlier_mask, col] = df[col].median()

        results["outliers"] = f"Applied {method} method to {len(columns)} columns"

    results["shape_change"] = f"{original_shape} → {df.shape}"
    return df, results

@tool
def perform_data_cleaning(path: str) -> Dict[str, Any]:
    """
    Analyzes dataset for data quality issues and provides cleaning recommendations.
    User just says 'perform cleaning' or 'clean the data'.
    """
    from helpers import get_dataset
    df = get_dataset(path)
    if df is None:
        return {"error": "Dataset not found. Please load dataset first."}

    # Detect issues internally
    issues = _detect_data_issues(df)

    if not issues:
        return {
            "status": "clean",
            "message": "🎉 Great! Your dataset is already clean - no major data quality issues detected.",
            "dataset_info": f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns"
        }

    # Generate recommendations
    recommendations = {}
    alternatives = {}

    if "missing_values" in issues:
        missing_info = issues["missing_values"]
        recommendations["missing_values"] = {
            "method": "fill_mean",
            "reason": "Preserves all data while maintaining statistical properties",
            "affected": f"{missing_info['total']} missing values in {len(missing_info['columns'])} columns"
        }
        alternatives["missing_values"] = [
            "fill_median (robust to outliers)",
            "drop (removes incomplete records)",
            "fill_mode (most frequent value)"
        ]

    if "duplicates" in issues:
        dup_info = issues["duplicates"]
        recommendations["duplicates"] = {
            "method": "remove_first",
            "reason": "Keeps earliest occurrence, standard approach",
            "affected": f"{dup_info['count']} duplicate rows ({dup_info['percentage']}%)"
        }
        alternatives["duplicates"] = [
            "remove_last (keep most recent)",
            "keep_all (no removal)"
        ]

    if "outliers" in issues:
        outlier_info = issues["outliers"]["columns"]
        outlier_cols = list(outlier_info.keys())
        total_outliers = sum(col_info["count"] for col_info in outlier_info.values())

        recommendations["outliers"] = {
            "method": "cap",
            "reason": "Reduces outlier impact while preserving all data",
            "affected": f"{total_outliers} outliers across {len(outlier_cols)} columns"
        }
        alternatives["outliers"] = [
            "remove (delete outlier rows)",
            "replace_median (substitute with median)",
            "replace_mean (substitute with mean)"
        ]

    # Store analysis for later use
    PIPELINE_STATE["cleaning_analysis"] = {
        "issues": issues,
        "recommendations": recommendations,
        "alternatives": alternatives
    }

    # Format response
    response = "🔍 **Data Quality Analysis Complete**\n\n"
    response += f"**Dataset:** {df.shape[0]} rows, {df.shape[1]} columns\n"
    response += f"**Issues Found:** {len(issues)} types of data quality issues\n\n"

    response += "**🎯 RECOMMENDED CLEANING APPROACH:**\n"
    for issue_type, rec in recommendations.items():
        response += f"• **{issue_type.replace('_', ' ').title()}:** {rec['method']} - {rec['reason']}\n"
        response += f"  ↳ {rec['affected']}\n"

    response += "\n**🔄 ALTERNATIVE OPTIONS AVAILABLE:**\n"
    for issue_type, alts in alternatives.items():
        response += f"• **{issue_type.replace('_', ' ').title()}:** {', '.join(alts)}\n"

    response += "\n**💡 NEXT STEPS:**\n"
    response += "• Type 'proceed with cleaning' to use recommended approach\n"
    response += "• Type 'modify [issue_type]' to choose different method for specific issues\n"
    response += "• Type 'show details' for more information about detected issues"

    # Collect detailed data for frontend
    detailed_data = {
        "issues_analysis": {},
        "cleaning_impact": {
            "total_rows": int(df.shape[0]),
            "total_columns": int(df.shape[1]),
            "issues_count": len(issues)
        },
        "before_stats": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
        "data_preview": df.head().to_dict('records')
    }

    # Detailed analysis for each issue type
    if "missing_values" in issues:
        missing_info = issues["missing_values"]
        detailed_data["issues_analysis"]["missing_values"] = {
            "affected_columns": missing_info["columns"],
            "column_wise_count": {k: float(v) for k, v in missing_info["percentages"].items()},
            "total_missing": int(missing_info["total"]),
            "severity": "High" if missing_info["total"] > len(df) * 0.1 else "Medium" if missing_info["total"] > 0 else "Low",
            "pattern": "Systematic" if len(missing_info["columns"]) < df.shape[1] * 0.3 else "Random"
        }

    if "duplicates" in issues:
        dup_info = issues["duplicates"]
        # Get sample duplicate rows
        duplicate_mask = df.duplicated(keep=False)
        sample_duplicates = df[duplicate_mask].head(3).to_dict('records') if duplicate_mask.any() else []
        
        detailed_data["issues_analysis"]["duplicates"] = {
            "count": int(dup_info["count"]),
            "percentage": float(dup_info["percentage"]),
            "sample_duplicates": sample_duplicates,
            "severity": "High" if dup_info["percentage"] > 5 else "Medium" if dup_info["percentage"] > 1 else "Low"
        }

    if "outliers" in issues:
        outlier_info = issues["outliers"]["columns"]
        outlier_details = {}
        outlier_samples = {}
        
        for col, col_info in outlier_info.items():
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_details[col] = {
                "count": int(col_info["count"]),
                "percentage": float(col_info["percentage"]),
                "bounds": {
                    "lower": float(lower_bound),
                    "upper": float(upper_bound),
                    "Q1": float(Q1),
                    "Q3": float(Q3),
                    "IQR": float(IQR)
                }
            }
            
            # Get sample outlier values
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_values = df[outlier_mask][col].head(3).tolist()
            outlier_samples[col] = [float(x) for x in outlier_values]
        
        total_outliers = sum(col_info["count"] for col_info in outlier_info.values())
        detailed_data["issues_analysis"]["outliers"] = {
            "column_wise_outliers": outlier_details,
            "outlier_samples": outlier_samples,
            "total_outliers": int(total_outliers),
            "affected_columns": len(outlier_info),
            "severity": "High" if total_outliers > len(df) * 0.1 else "Medium" if total_outliers > len(df) * 0.05 else "Low"
        }

    # Store detailed data for frontend
    PIPELINE_STATE["last_detailed_data"] = detailed_data

    return {
        "analysis_complete": True,
        "issues_found": len(issues),
        "formatted_response": response,
        "recommendations": recommendations,
        "alternatives": alternatives
    }

@tool
def execute_data_cleaning(path: str, approach: str = "recommended", modifications: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Executes data cleaning based on user's choice.
    User says 'proceed with cleaning' or 'modify outliers to remove'.
    """
    from helpers import get_dataset
    df = get_dataset(path)
    if df is None:
        return {"error": "Dataset not found"}

    if "cleaning_analysis" not in PIPELINE_STATE:
        return {"error": "Please run data analysis first using 'perform cleaning'"}

    df = df.copy()
    analysis = PIPELINE_STATE["cleaning_analysis"]

    # Build cleaning plan
    cleaning_plan = {}

    if approach == "recommended":
        # Use all recommended methods
        for issue_type, rec in analysis["recommendations"].items():
            if issue_type == "missing_values":
                cleaning_plan["missing_values"] = {
                    "method": rec["method"],
                    "columns": None  # Apply to all columns with missing values
                }
            elif issue_type == "duplicates":
                cleaning_plan["duplicates"] = {"method": rec["method"]}
            elif issue_type == "outliers":
                outlier_cols = list(analysis["issues"]["outliers"]["columns"].keys())
                cleaning_plan["outliers"] = {
                    "method": rec["method"],
                    "columns": outlier_cols
                }

    # Apply any modifications
    if modifications:
        for issue_type, new_method in modifications.items():
            if issue_type in cleaning_plan:
                cleaning_plan[issue_type]["method"] = new_method

    # Execute cleaning
    cleaned_df, results = _execute_cleaning(df, cleaning_plan)

    # Update cache with correct key
    from helpers import get_cache_key
    cache_key = get_cache_key(path)
    if cache_key:
        DATAFRAME_CACHE[cache_key] = cleaned_df
    else:
        DATAFRAME_CACHE[path] = cleaned_df
    PIPELINE_STATE["completed_operations"].append("data_cleaning")

    # Format response
    response = "✅ **Data Cleaning Complete!**\n\n"
    response += f"**Shape Change:** {results['shape_change']}\n\n"
    response += "**Operations Applied:**\n"
    for operation, description in results.items():
        if operation != "shape_change":
            response += f"• {operation.replace('_', ' ').title()}: {description}\n"

    response += "\n**🎯 Your dataset is now clean and ready for:**\n"
    response += "• Feature engineering ('perform feature engineering')\n"
    response += "• Direct model training ('train model')\n"
    response += "• Data exploration ('explore data')"

    # Collect detailed data for frontend
    detailed_data = {
        "cleaning_summary": {
            "operations_performed": list(results.keys()),
            "rows_before": int(df.shape[0]),
            "rows_after": int(cleaned_df.shape[0]),
            "rows_removed": int(df.shape[0] - cleaned_df.shape[0]),
            "columns_affected": len(cleaning_plan)
        },
        "before_after_comparison": {
            "shape_before": df.shape,
            "shape_after": cleaned_df.shape,
            "quality_before": {
                "missing_values": int(df.isnull().sum().sum()),
                "duplicates": int(df.duplicated().sum()),
                "total_rows": int(df.shape[0])
            },
            "quality_after": {
                "missing_values": int(cleaned_df.isnull().sum().sum()),
                "duplicates": int(cleaned_df.duplicated().sum()),
                "total_rows": int(cleaned_df.shape[0])
            }
        },
        "transformation_log": [],
        "cleaned_data_preview": cleaned_df.head().to_dict('records'),
        "quality_improvement": {}
    }

    # Build transformation log
    for operation, description in results.items():
        if operation != "shape_change":
            detailed_data["transformation_log"].append({
                "operation": operation.replace('_', ' ').title(),
                "description": description,
                "impact": "Data quality improved"
            })

    # Calculate quality improvements
    before_quality = detailed_data["before_after_comparison"]["quality_before"]
    after_quality = detailed_data["before_after_comparison"]["quality_after"]
    
    detailed_data["quality_improvement"] = {
        "missing_values_reduced": int(before_quality["missing_values"] - after_quality["missing_values"]),
        "duplicates_removed": int(before_quality["duplicates"] - after_quality["duplicates"]),
        "data_retention_rate": round((after_quality["total_rows"] / before_quality["total_rows"]) * 100, 1)
    }

    # Add before/after statistics comparison
    if len(cleaned_df.select_dtypes(include=['number']).columns) > 0:
        detailed_data["stats_comparison"] = {
            "before": df.describe().to_dict(),
            "after": cleaned_df.describe().to_dict()
        }

    # Store detailed data for frontend
    PIPELINE_STATE["last_detailed_data"] = detailed_data

    return {
        "success": True,
        "cleaning_completed": True,
        "operations_applied": list(results.keys()),
        "formatted_response": response,
        "new_shape": cleaned_df.shape
    }
