import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from helpers import DATAFRAME_CACHE, PIPELINE_STATE, _analyze_features

def _execute_feature_engineering(df: pd.DataFrame, engineering_plan: Dict[str, Any]) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Internal function to execute feature engineering operations."""
    results = {}

    # Handle encoding
    if "encoding" in engineering_plan:
        method = engineering_plan["encoding"]["method"]
        columns = engineering_plan["encoding"]["columns"]

        if method == "one_hot":
            df = pd.get_dummies(df, columns=columns, drop_first=True)
            results["encoding"] = f"Applied one-hot encoding to {len(columns)} columns"
        elif method == "label":
            le = LabelEncoder()
            for col in columns:
                if col in df.columns:
                    df[col] = le.fit_transform(df[col].astype(str))
            results["encoding"] = f"Applied label encoding to {len(columns)} columns"

    # Handle scaling
    if "scaling" in engineering_plan:
        method = engineering_plan["scaling"]["method"]
        columns = engineering_plan["scaling"]["columns"]

        if method == "standard":
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
            results["scaling"] = f"Applied standard scaling to {len(columns)} columns"
        elif method == "minmax":
            scaler = MinMaxScaler()
            df[columns] = scaler.fit_transform(df[columns])
            results["scaling"] = f"Applied min-max scaling to {len(columns)} columns"

    # Handle correlation/dimensionality reduction
    if "correlation" in engineering_plan:
        method = engineering_plan["correlation"]["method"]
        
        if method == "pca":
            # Apply PCA to numeric features
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 1:
                # Scale first for PCA
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[numeric_cols])
                
                # Apply PCA (keep 95% variance)
                pca = PCA(n_components=0.95)
                pca_data = pca.fit_transform(scaled_data)
                
                # Create new dataframe with PCA components
                pca_cols = [f'PC{i+1}' for i in range(pca_data.shape[1])]
                pca_df = pd.DataFrame(pca_data, columns=pca_cols, index=df.index)
                
                # Replace numeric columns with PCA components
                df = df.drop(columns=numeric_cols)
                df = pd.concat([df, pca_df], axis=1)
                
                results["correlation"] = f"Applied PCA: {len(numeric_cols)} features → {len(pca_cols)} components (95% variance)"
        
        elif method == "remove_redundant":
            # Remove highly correlated features
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr().abs()
                upper_tri = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
                df = df.drop(columns=to_drop)
                results["correlation"] = f"Removed {len(to_drop)} highly correlated features"

    return df, results

@tool
def perform_feature_engineering(path: str) -> Dict[str, Any]:
    """
    Analyzes dataset features and provides engineering recommendations.
    User says 'perform feature engineering' or 'engineer features'.
    """
    from helpers import get_dataset
    df = get_dataset(path)
    if df is None:
        return {"error": "Dataset not found"}

    # Analyze features internally
    analysis = _analyze_features(df)

    # Generate recommendations
    recommendations = {}
    alternatives = {}

    if analysis["categorical_features"]:
        cat_features = analysis["categorical_features"]
        recommendations["encoding"] = {
            "method": "one_hot",
            "reason": "Creates binary features, works well with most algorithms",
            "affected": f"{len(cat_features)} categorical columns: {', '.join(cat_features[:3])}{'...' if len(cat_features) > 3 else ''}"
        }
        alternatives["encoding"] = [
            "label (ordinal numbering)",
            "target (based on target variable)",
            "frequency (based on value counts)"
        ]

    if analysis["needs_scaling"]:
        scale_features = analysis["needs_scaling"]
        recommendations["scaling"] = {
            "method": "standard",
            "reason": "Normalizes features to mean=0, std=1, works well with most ML algorithms",
            "affected": f"{len(scale_features)} numeric columns with large ranges"
        }
        alternatives["scaling"] = [
            "minmax (scale to 0-1 range)",
            "robust (uses median, robust to outliers)",
            "normalize (unit vector scaling)"
        ]

    if analysis["high_correlations"]:
        corr_pairs = analysis["high_correlations"]
        recommendations["correlation"] = {
            "method": "remove_redundant",
            "reason": "Removes highly correlated features to prevent multicollinearity",
            "affected": f"{len(corr_pairs)} highly correlated feature pairs"
        }
        alternatives["correlation"] = [
            "keep_all (no removal)",
            "pca (principal component analysis)",
            "manual_select (choose which to keep)"
        ]

    if not recommendations:
        return {
            "status": "ready",
            "message": "🎉 Your features are already well-prepared for machine learning!",
            "feature_info": f"Dataset: {len(analysis['numeric_features'])} numeric, {len(analysis['categorical_features'])} categorical features"
        }

    # Store analysis
    PIPELINE_STATE["feature_analysis"] = {
        "analysis": analysis,
        "recommendations": recommendations,
        "alternatives": alternatives
    }

    # Format response
    response = "🔧 **Feature Engineering Analysis Complete**\n\n"
    response += f"**Dataset:** {df.shape[0]} rows, {df.shape[1]} features\n"
    response += f"**Feature Types:** {len(analysis['numeric_features'])} numeric, {len(analysis['categorical_features'])} categorical\n\n"

    response += "**🎯 RECOMMENDED FEATURE ENGINEERING:**\n"
    for eng_type, rec in recommendations.items():
        response += f"• **{eng_type.title()}:** {rec['method']} - {rec['reason']}\n"
        response += f"  ↳ {rec['affected']}\n"

    response += "\n**🔄 ALTERNATIVE OPTIONS:**\n"
    for eng_type, alts in alternatives.items():
        response += f"• **{eng_type.title()}:** {', '.join(alts)}\n"

    response += "\n**💡 NEXT STEPS:**\n"
    response += "• Type 'proceed with feature engineering' to apply recommendations\n"
    response += "• Type 'modify [type]' to choose different methods\n"
    response += "• Type 'show feature details' for more information"

    # Collect detailed data for frontend
    numeric_features = analysis["numeric_features"]
    categorical_features = analysis["categorical_features"]
    
    # Calculate correlation matrix for numeric features
    correlation_matrix = {}
    if len(numeric_features) > 1:
        corr_df = df[numeric_features].corr()
        correlation_matrix = corr_df.to_dict()
    
    # Get sample values for categorical features
    categorical_samples = {}
    for col in categorical_features[:5]:  # Limit to first 5 for performance
        categorical_samples[col] = df[col].value_counts().head(5).to_dict()
    
    # Analyze scaling requirements
    scaling_analysis = {}
    for col in numeric_features:
        scaling_analysis[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "range": float(df[col].max() - df[col].min()),
            "needs_scaling": col in analysis["needs_scaling"]
        }
    
    # Feature relationship analysis
    high_corr_pairs = []
    if analysis["high_correlations"]:
        for pair in analysis["high_correlations"][:10]:  # Limit to top 10
            high_corr_pairs.append({
                "feature1": pair[0],
                "feature2": pair[1], 
                "correlation": float(pair[2])
            })
    
    detailed_data = {
        "feature_analysis": {
            "numeric_features": {
                "names": numeric_features,
                "count": len(numeric_features),
                "scaling_needed": analysis["needs_scaling"],
                "scaling_analysis": scaling_analysis,
                "correlation_matrix": correlation_matrix
            },
            "categorical_features": {
                "names": categorical_features,
                "count": len(categorical_features),
                "high_cardinality": analysis["high_cardinality"],
                "sample_values": categorical_samples,
                "encoding_needed": categorical_features
            },
            "feature_relationships": {
                "high_correlations": high_corr_pairs,
                "total_correlations": len(analysis["high_correlations"]) if analysis["high_correlations"] else 0
            }
        },
        "engineering_opportunities": {
            "scaling_candidates": {
                "count": len(analysis["needs_scaling"]),
                "features": analysis["needs_scaling"],
                "reason": "Features with large ranges or different scales"
            },
            "encoding_candidates": {
                "count": len(categorical_features),
                "features": categorical_features,
                "reason": "Categorical features need numeric encoding"
            },
            "correlation_reduction": {
                "count": len(analysis["high_correlations"]) if analysis["high_correlations"] else 0,
                "pairs": high_corr_pairs[:5],
                "reason": "Highly correlated features may cause multicollinearity"
            }
        },
        "current_feature_stats": df.describe().to_dict() if len(numeric_features) > 0 else {},
        "data_preview": df.head().to_dict('records'),
        "feature_summary": {
            "total_features": int(df.shape[1]),
            "numeric_count": len(numeric_features),
            "categorical_count": len(categorical_features),
            "engineering_needed": len(recommendations) > 0
        }
    }

    # Store detailed data for frontend
    PIPELINE_STATE["last_detailed_data"] = detailed_data

    return {
        "analysis_complete": True,
        "engineering_needed": len(recommendations),
        "formatted_response": response,
        "recommendations": recommendations
    }

@tool
def execute_feature_engineering(path: str, approach: str = "recommended", modifications: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Executes feature engineering based on user's choice.
    """
    from helpers import get_dataset
    df = get_dataset(path)
    if df is None:
        return {"error": "Dataset not found"}

    if "feature_analysis" not in PIPELINE_STATE:
        return {"error": "Please run feature analysis first using 'perform feature engineering'"}

    df = df.copy()
    analysis = PIPELINE_STATE["feature_analysis"]

    # Build engineering plan
    engineering_plan = {}

    if approach == "recommended":
        for eng_type, rec in analysis["recommendations"].items():
            if eng_type == "encoding":
                engineering_plan["encoding"] = {
                    "method": "one_hot" if rec["method"] == "one_hot" else "label",
                    "columns": analysis["analysis"]["categorical_features"]
                }
            elif eng_type == "scaling":
                engineering_plan["scaling"] = {
                    "method": "standard" if rec["method"] == "standard" else "minmax",
                    "columns": analysis["analysis"]["needs_scaling"]
                }
            elif eng_type == "correlation":
                engineering_plan["correlation"] = {
                    "method": rec["method"]
                }

    # Apply modifications
    if modifications:
        for eng_type, new_method in modifications.items():
            if eng_type in engineering_plan:
                engineering_plan[eng_type]["method"] = new_method

    # Execute engineering
    engineered_df, results = _execute_feature_engineering(df, engineering_plan)

    # Update cache with correct key
    from helpers import get_cache_key
    cache_key = get_cache_key(path)
    if cache_key:
        DATAFRAME_CACHE[cache_key] = engineered_df
    else:
        DATAFRAME_CACHE[path] = engineered_df
    PIPELINE_STATE["completed_operations"].append("feature_engineering")

    response = "✅ **Feature Engineering Complete!**\n\n"
    response += f"**Shape Change:** {df.shape} → {engineered_df.shape}\n\n"
    response += "**Transformations Applied:**\n"
    for operation, description in results.items():
        response += f"• {description}\n"

    response += "\n**🎯 Your features are now ready for:**\n"
    response += "• Model training ('train model')\n"
    response += "• Algorithm selection ('suggest algorithms')\n"
    response += "• Cross-validation ('validate model')"

    # Collect detailed data for frontend
    original_numeric = df.select_dtypes(include=['number']).columns.tolist()
    original_categorical = df.select_dtypes(include=['object']).columns.tolist()
    
    engineered_numeric = engineered_df.select_dtypes(include=['number']).columns.tolist()
    engineered_categorical = engineered_df.select_dtypes(include=['object']).columns.tolist()
    
    # Calculate transformation impact
    features_added = set(engineered_df.columns) - set(df.columns)
    features_removed = set(df.columns) - set(engineered_df.columns)
    features_modified = set(df.columns) & set(engineered_df.columns)
    
    # Build transformation log
    transformation_log = []
    for operation, description in results.items():
        transformation_log.append({
            "operation": operation.title(),
            "description": description,
            "impact": "Feature transformation applied"
        })
    
    # Calculate correlation improvement (if applicable)
    correlation_improvement = {}
    if len(original_numeric) > 1 and len(engineered_numeric) > 1:
        try:
            original_corr = df[original_numeric].corr().abs()
            engineered_corr = engineered_df[engineered_numeric].corr().abs()
            
            # Count high correlations (>0.8)
            original_high_corr = (original_corr > 0.8).sum().sum() - len(original_numeric)  # Subtract diagonal
            engineered_high_corr = (engineered_corr > 0.8).sum().sum() - len(engineered_numeric)
            
            correlation_improvement = {
                "high_correlations_before": int(original_high_corr),
                "high_correlations_after": int(engineered_high_corr),
                "improvement": int(original_high_corr - engineered_high_corr)
            }
        except:
            correlation_improvement = {"error": "Could not calculate correlation improvement"}
    
    detailed_data = {
        "transformation_summary": {
            "features_before": int(df.shape[1]),
            "features_after": int(engineered_df.shape[1]),
            "features_added": list(features_added),
            "features_removed": list(features_removed),
            "features_modified": list(features_modified),
            "operations_count": len(results)
        },
        "before_after_comparison": {
            "shape_before": [int(df.shape[0]), int(df.shape[1])],
            "shape_after": [int(engineered_df.shape[0]), int(engineered_df.shape[1])],
            "numeric_features_before": {
                "count": len(original_numeric),
                "names": original_numeric
            },
            "numeric_features_after": {
                "count": len(engineered_numeric),
                "names": engineered_numeric
            },
            "categorical_features_before": {
                "count": len(original_categorical),
                "names": original_categorical
            },
            "categorical_features_after": {
                "count": len(engineered_categorical),
                "names": engineered_categorical
            }
        },
        "feature_statistics": {
            "before_stats": df.describe().to_dict() if len(original_numeric) > 0 else {},
            "after_stats": engineered_df.describe().to_dict() if len(engineered_numeric) > 0 else {}
        },
        "transformation_log": transformation_log,
        "engineered_data_preview": engineered_df.head().to_dict('records'),
        "correlation_improvement": correlation_improvement,
        "quality_metrics": {
            "feature_count_change": int(engineered_df.shape[1] - df.shape[1]),
            "numeric_feature_change": len(engineered_numeric) - len(original_numeric),
            "categorical_feature_change": len(engineered_categorical) - len(original_categorical),
            "data_retention_rate": 100.0  # Feature engineering typically doesn't remove rows
        }
    }

    # Store detailed data for frontend
    PIPELINE_STATE["last_detailed_data"] = detailed_data

    return {
        "success": True,
        "engineering_completed": True,
        "transformations": list(results.keys()),
        "formatted_response": response,
        "new_shape": engineered_df.shape
    }
