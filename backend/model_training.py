import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from helpers import DATAFRAME_CACHE, PIPELINE_STATE, get_dataset
from config import Config

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


@tool
def execute_model_training(path: str, algorithm: str = "recommended") -> Dict[str, Any]:
    """
    Executes model training based on user's algorithm choice.
    User says 'proceed with training random_forest' or 'proceed with training recommended'.
    """
    df = get_dataset(path)
    if df is None:
        return {"error": "Dataset not found"}

    if "training_analysis" not in PIPELINE_STATE:
        return {"error": "Please run problem detection first using 'train model'"}

    analysis = PIPELINE_STATE["training_analysis"]
    problem_type = analysis["problem_type"]
    target_column = analysis["target_column"]
    available_algorithms = analysis["available_algorithms"]
    recommended_algorithms = analysis["recommended_algorithms"]
    
    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE
    )
    
    # Define algorithm implementations
    algorithm_implementations = {}
    if problem_type == "classification":
        algorithm_implementations = {
            "random_forest": RandomForestClassifier(random_state=Config.RANDOM_STATE),
            "gradient_boosting": GradientBoostingClassifier(random_state=Config.RANDOM_STATE),
            "logistic_regression": LogisticRegression(random_state=Config.RANDOM_STATE, max_iter=Config.MAX_ITER),
            "svm": SVC(random_state=Config.RANDOM_STATE),
            "knn": KNeighborsClassifier()
        }
        metric_name = "accuracy"
    else:  # regression
        algorithm_implementations = {
            "random_forest": RandomForestRegressor(random_state=Config.RANDOM_STATE),
            "gradient_boosting": GradientBoostingRegressor(random_state=Config.RANDOM_STATE),
            "linear_regression": LinearRegression(),
            "ridge_regression": Ridge(random_state=Config.RANDOM_STATE),
            "lasso_regression": Lasso(random_state=Config.RANDOM_STATE),
            "svr": SVR(),
            "knn": KNeighborsRegressor()
        }
        metric_name = "r2_score"
    
    # Determine which algorithms to train
    algorithms_to_train = {}
    
    if algorithm == "recommended":
        for alg_name in recommended_algorithms:
            if alg_name in algorithm_implementations:
                algorithms_to_train[alg_name] = algorithm_implementations[alg_name]
    elif algorithm == "all":
        algorithms_to_train = {k: v for k, v in algorithm_implementations.items() if k in available_algorithms}
    else:
        # Train specific algorithm
        if algorithm in algorithm_implementations and algorithm in available_algorithms:
            algorithms_to_train[algorithm] = algorithm_implementations[algorithm]
        else:
            return {
                "error": f"Algorithm '{algorithm}' not available for {problem_type}",
                "available": list(available_algorithms.keys())
            }
    
    if not algorithms_to_train:
        return {"error": "No algorithms selected for training"}

    # Train selected algorithms
    results = {}
    for alg_name, model in algorithms_to_train.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if problem_type == "classification":
                score = accuracy_score(y_test, y_pred)
            else:
                score = r2_score(y_test, y_pred)

            results[alg_name] = {
                "model": model,
                "score": score,
                "predictions": y_pred
            }
        except Exception as e:
            results[alg_name] = {"error": str(e)}

    # Find best model
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        best_algorithm = max(valid_results.keys(), key=lambda k: valid_results[k]["score"])
        best_score = valid_results[best_algorithm]["score"]
    else:
        return {"error": "All algorithms failed to train"}

    # Store results (without raw data objects to avoid serialization issues)
    algorithms_summary = {}
    for alg_name, result in valid_results.items():
        if "error" not in result:
            algorithms_summary[alg_name] = {
                "score": float(result["score"]),
                "algorithm_name": alg_name
            }
    
    PIPELINE_STATE["models_trained"] = {
        "target_column": str(target_column),
        "problem_type": str(problem_type),
        "algorithms": algorithms_summary,
        "best_algorithm": str(best_algorithm),
        "best_score": float(best_score),
        "test_samples": int(len(X_test)),
        "feature_count": int(X.shape[1])
    }
    
    PIPELINE_STATE["completed_operations"].append("model_training")

    # Format response
    response = "✅ **Model Training Complete!**\n\n"
    response += f"**Problem Type:** {problem_type.title()}\n"
    response += f"**Target Column:** {target_column}\n"
    response += f"**Training Data:** {len(X_train)} samples, {X.shape[1]} features\n\n"

    response += "**🏆 MODEL PERFORMANCE:**\n"
    for alg_name, result in valid_results.items():
        score = result["score"]
        status = "👑 BEST" if alg_name == best_algorithm else ""
        response += f"• **{alg_name.replace('_', ' ').title()}:** {score:.4f} {metric_name} {status}\n"

    response += f"\n**🎯 RECOMMENDED MODEL:** {best_algorithm.replace('_', ' ').title()} ({best_score:.4f} {metric_name})\n\n"
    
    # Performance interpretation
    if problem_type == "classification":
        if best_score >= 0.90:
            performance = "Excellent 🌟"
        elif best_score >= 0.80:
            performance = "Very Good 👍"
        elif best_score >= 0.70:
            performance = "Good ✓"
        else:
            performance = "Needs Improvement 🔧"
    else:  # regression
        if best_score >= 0.90:
            performance = "Excellent 🌟"
        elif best_score >= 0.80:
            performance = "Very Good 👍"
        elif best_score >= 0.60:
            performance = "Good ✓"
        else:
            performance = "Needs Improvement 🔧"
    
    response += f"**📊 PERFORMANCE LEVEL:** {performance}\n\n"
    response += "**💡 NEXT STEPS:**\n"
    response += "• Type 'evaluate model' for detailed performance analysis\n"
    response += "• Type 'compare models' to see all model comparisons\n"
    response += "• Type 'improve model' for suggestions to boost performance\n"

    # Collect detailed data for frontend
    import time
    import numpy as np
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
    
    # Get the best model for detailed analysis
    best_model = valid_results[best_algorithm]["model"]
    y_pred_best = valid_results[best_algorithm]["predictions"]
    
    # Calculate cross-validation scores
    try:
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy' if problem_type == 'classification' else 'r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    except:
        cv_scores = [best_score] * 5
        cv_mean = best_score
        cv_std = 0.0
    
    # Performance metrics
    performance_metrics = {
        "primary_metric": {
            "name": str(metric_name),
            "value": float(best_score),
            "interpretation": str(performance.split()[0].lower())
        }
    }
    
    if problem_type == "classification":
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        try:
            # Handle multiclass vs binary
            average_method = 'weighted' if len(np.unique(y_test)) > 2 else 'binary'
            
            precision = precision_score(y_test, y_pred_best, average=average_method, zero_division=0)
            recall = recall_score(y_test, y_pred_best, average=average_method, zero_division=0)
            f1 = f1_score(y_test, y_pred_best, average=average_method, zero_division=0)
            
            performance_metrics["detailed_metrics"] = {
                "accuracy": float(best_score),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }
            
            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2:
                try:
                    if hasattr(best_model, "predict_proba"):
                        y_proba = best_model.predict_proba(X_test)[:, 1]
                        roc_auc = roc_auc_score(y_test, y_proba)
                        performance_metrics["detailed_metrics"]["roc_auc"] = float(roc_auc)
                except:
                    pass
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred_best)
            performance_metrics["confusion_matrix"] = cm.tolist()
            
        except Exception as e:
            print(f"Error calculating classification metrics: {e}")
            performance_metrics["detailed_metrics"] = {"accuracy": float(best_score)}
    
    else:  # regression
        try:
            mse = mean_squared_error(y_test, y_pred_best)
            mae = mean_absolute_error(y_test, y_pred_best)
            rmse = np.sqrt(mse)
            
            performance_metrics["detailed_metrics"] = {
                "r2_score": float(best_score),
                "mean_squared_error": float(mse),
                "root_mean_squared_error": float(rmse),
                "mean_absolute_error": float(mae)
            }
        except Exception as e:
            print(f"Error calculating regression metrics: {e}")
            performance_metrics["detailed_metrics"] = {"r2_score": float(best_score)}
    
    # Cross-validation results
    performance_metrics["cross_validation"] = {
        "mean_score": float(cv_mean),
        "std_score": float(cv_std),
        "scores": [float(score) for score in cv_scores],
        "confidence_interval": [float(cv_mean - 2*cv_std), float(cv_mean + 2*cv_std)]
    }
    
    # Feature importance (if available)
    feature_importance_data = {}
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_names = X.columns.tolist()
        
        # Sort features by importance
        importance_pairs = list(zip(feature_names, importances))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        feature_importance_data = {
            "available": True,
            "top_features": [
                {
                    "name": name,
                    "importance": float(importance),
                    "rank": idx + 1,
                    "percentage": float(importance / sum(importances) * 100)
                }
                for idx, (name, importance) in enumerate(importance_pairs[:10])
            ],
            "total_features": len(feature_names)
        }
    elif hasattr(best_model, 'coef_'):
        # For linear models, use coefficient magnitudes
        if problem_type == "classification" and len(best_model.coef_.shape) > 1:
            # Multiclass - use mean of absolute coefficients
            coef_importance = np.mean(np.abs(best_model.coef_), axis=0)
        else:
            # Binary classification or regression
            coef_importance = np.abs(best_model.coef_.flatten())
        
        feature_names = X.columns.tolist()
        importance_pairs = list(zip(feature_names, coef_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        feature_importance_data = {
            "available": True,
            "type": "coefficients",
            "top_features": [
                {
                    "name": name,
                    "importance": float(importance),
                    "rank": idx + 1,
                    "percentage": float(importance / sum(coef_importance) * 100)
                }
                for idx, (name, importance) in enumerate(importance_pairs[:10])
            ],
            "total_features": len(feature_names)
        }
    else:
        feature_importance_data = {"available": False, "reason": "Model does not provide feature importance"}
    
    # Training summary
    training_summary = {
        "algorithm_used": str(best_algorithm),
        "algorithm_display_name": str(best_algorithm.replace('_', ' ').title()),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "target_column": str(target_column),
        "problem_type": str(problem_type),
        "cross_validation_performed": True,
        "model_complexity": str("high" if best_algorithm in ["random_forest", "gradient_boosting"] else "medium" if best_algorithm in ["svm", "knn"] else "low")
    }
    
    # Get model hyperparameters
    try:
        hyperparameters = best_model.get_params()
        # Filter out complex objects and keep only basic parameters
        simple_params = {}
        for key, value in hyperparameters.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                # Convert numpy types to native Python types
                if isinstance(value, (np.integer, np.floating)):
                    simple_params[str(key)] = float(value) if isinstance(value, np.floating) else int(value)
                else:
                    simple_params[str(key)] = value
        training_summary["hyperparameters_used"] = simple_params
    except:
        training_summary["hyperparameters_used"] = {}
    
    # Model diagnostics
    model_diagnostics = {
        "overfitting_check": {
            "train_score": float(cv_mean),
            "test_score": float(best_score),
            "difference": float(abs(cv_mean - best_score)),
            "status": str("good" if abs(cv_mean - best_score) < 0.1 else "potential_overfitting")
        },
        "prediction_confidence": {
            "high_confidence_predictions": 0,
            "low_confidence_predictions": 0
        }
    }
    
    # Calculate prediction confidence if possible
    if hasattr(best_model, "predict_proba") and problem_type == "classification":
        try:
            probabilities = best_model.predict_proba(X_test)
            max_probabilities = np.max(probabilities, axis=1)
            model_diagnostics["prediction_confidence"] = {
                "mean_confidence": float(np.mean(max_probabilities)),
                "high_confidence_predictions": int(np.sum(max_probabilities > 0.8)),
                "low_confidence_predictions": int(np.sum(max_probabilities < 0.6)),
                "confidence_distribution": {
                    "very_high": int(np.sum(max_probabilities > 0.9)),
                    "high": int(np.sum((max_probabilities > 0.8) & (max_probabilities <= 0.9))),
                    "medium": int(np.sum((max_probabilities > 0.6) & (max_probabilities <= 0.8))),
                    "low": int(np.sum(max_probabilities <= 0.6))
                }
            }
        except:
            pass
    
    detailed_data = {
        "training_summary": training_summary,
        "performance_metrics": performance_metrics,
        "feature_importance": feature_importance_data,
        "model_diagnostics": model_diagnostics,
        "predictions_sample": {
            "actual_values": [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in (y_test.head(10).tolist() if hasattr(y_test, 'head') else y_test[:10].tolist())],
            "predicted_values": [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in y_pred_best[:10].tolist()],
            "feature_names": [str(x) for x in X.columns.tolist()]
        },
        "training_data_info": {
            "total_samples": int(len(df)),
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "feature_count": int(X.shape[1]),
            # "target_distribution": {str(k): int(v) for k, v in target_series.value_counts().head(10).to_dict().items()} if problem_type == "classification" else {}
            "target_distribution": {str(k): int(v) for k, v in y.value_counts().head(10).to_dict().items()} if problem_type == "classification" else {}

        }
    }

    # Store detailed data for frontend
    PIPELINE_STATE["last_detailed_data"] = detailed_data

    return {
        "success": True,
        "training_completed": True,
        "models_trained": len(valid_results),
        "best_algorithm": best_algorithm,
        "best_score": best_score,
        "performance_level": performance,
        "formatted_response": response,
        "problem_type": problem_type
    }

@tool
def train_model(path: str, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes dataset to detect problem type and suggests available models.
    User says 'train model' or 'train the model' to start the training process.
    """
    print(f"🔧 DEBUG train_model called with path: '{path}'")
    print(f"🔧 DEBUG Available datasets: {list(DATAFRAME_CACHE.keys())}")
    
    df = get_dataset(path)
    if df is None:
        return {
            "error": "Dataset not found",
            "available_datasets": list(DATAFRAME_CACHE.keys()),
            "formatted_response": f"❌ Dataset '{path}' not found. Available datasets: {list(DATAFRAME_CACHE.keys())}"
        }
    
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
    
    # Analyze target column to determine problem type
    target_series = df[target_column]
    
    # Basic statistics
    unique_values = target_series.nunique()
    total_values = len(target_series)
    unique_ratio = unique_values / total_values
    data_type = str(target_series.dtype)
    sample_values = target_series.dropna().unique()[:10].tolist()
    
    # Determine problem type
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
    if unique_ratio < 0.05:
        classification_score += 3
    elif unique_ratio < 0.1:
        classification_score += 2
    elif unique_ratio > 0.5:
        regression_score += 2
    
    # Determine final problem type
    if classification_score > regression_score:
        if unique_values <= 2:
            problem_type = "classification"
            subtype = "binary_classification"
        else:
            problem_type = "classification"
            subtype = "multiclass_classification"
    else:
        problem_type = "regression"
        subtype = "continuous_regression"
    
    confidence = max(classification_score, regression_score) / (classification_score + regression_score + 1)

    # Get available algorithms based on problem type
    available_algorithms = {}
    recommended_algorithms = []
    
    if problem_type == "classification":
        available_algorithms = {
            "random_forest": "Random Forest - Robust ensemble method, handles overfitting well",
            "gradient_boosting": "Gradient Boosting - High performance sequential learning",
            "logistic_regression": "Logistic Regression - Fast, interpretable, good baseline",
            "svm": "Support Vector Machine - Effective for high-dimensional data",
            "knn": "K-Nearest Neighbors - Simple, non-parametric method"
        }
        if subtype == "binary_classification":
            recommended_algorithms = ["logistic_regression", "random_forest", "gradient_boosting"]
        else:
            recommended_algorithms = ["random_forest", "gradient_boosting", "logistic_regression"]
    else:  # regression
        available_algorithms = {
            "random_forest": "Random Forest - Robust ensemble, handles non-linearity",
            "gradient_boosting": "Gradient Boosting - High performance sequential learning",
            "linear_regression": "Linear Regression - Simple, interpretable baseline",
            "ridge_regression": "Ridge Regression - Regularized, prevents overfitting",
            "lasso_regression": "Lasso Regression - Feature selection + regularization",
            "svr": "Support Vector Regression - Effective for complex patterns",
            "knn": "K-Nearest Neighbors - Non-parametric, local patterns"
        }
        recommended_algorithms = ["random_forest", "gradient_boosting", "linear_regression"]
    
    # Store analysis for next step
    PIPELINE_STATE["training_analysis"] = {
        "target_column": target_column,
        "problem_type": problem_type,
        "subtype": subtype,
        "confidence": confidence,
        "available_algorithms": available_algorithms,
        "recommended_algorithms": recommended_algorithms,
        "target_stats": {
            "unique_values": unique_values,
            "unique_ratio": unique_ratio,
            "data_type": data_type,
            "sample_values": sample_values
        }
    }
    
    # Format response
    response = "🔍 **Problem Type Detection & Model Selection**\n\n"
    response += f"**Target Column:** {target_column}\n"
    response += f"**Problem Type:** {problem_type.title()}\n"
    response += f"**Subtype:** {subtype.replace('_', ' ').title()}\n"
    response += f"**Confidence:** {confidence:.1%}\n\n"
    
    response += "**📊 TARGET ANALYSIS:**\n"
    response += f"• **Data Type:** {data_type}\n"
    response += f"• **Unique Values:** {unique_values:,} ({unique_ratio:.1%} of total)\n"
    response += f"• **Sample Values:** {', '.join(map(str, sample_values[:5]))}\n\n"
    
    response += "**🎯 RECOMMENDED ALGORITHMS:**\n"
    for alg in recommended_algorithms:
        response += f"• **{alg.replace('_', ' ').title()}** - {available_algorithms[alg].split(' - ')[1]}\n"
    
    response += "\n**🔄 ALL AVAILABLE ALGORITHMS:**\n"
    for alg_name, description in available_algorithms.items():
        status = "⭐ RECOMMENDED" if alg_name in recommended_algorithms else ""
        response += f"• **{alg_name.replace('_', ' ').title()}** - {description.split(' - ')[1]} {status}\n"
    
    response += "\n**💡 NEXT STEPS:**\n"
    response += "• Type 'proceed with training [algorithm_name]' to train specific model\n"
    response += "• Type 'proceed with training recommended' to train all recommended models\n"
    response += "• Type 'proceed with training all' to train all available models\n"
    response += "\n**Examples:**\n"
    response += "• 'proceed with training random_forest'\n"
    response += "• 'proceed with training recommended'\n"
    
    # Collect detailed data for frontend
    dataset_characteristics = {
        "sample_size": "large" if len(df) > 10000 else "medium" if len(df) > 1000 else "small",
        "feature_count": int(len(df.columns) - 1),
        "data_complexity": "non-linear" if problem_type == "classification" and unique_values > 2 else "linear",
        "missing_values": int(df.isnull().sum().sum()),
        "categorical_features": int(len(df.select_dtypes(include=['object']).columns)),
        "numeric_features": int(len(df.select_dtypes(include=['number']).columns) - 1)  # exclude target
    }
    
    # Add class balance for classification
    if problem_type == "classification":
        class_counts = target_series.value_counts()
        dataset_characteristics["class_balance"] = {
            "classes": [str(x) for x in class_counts.index.tolist()],
            "counts": [int(x) for x in class_counts.values.tolist()],
            "percentages": [float(x) for x in (class_counts / len(target_series) * 100).round(2).tolist()],
            "is_balanced": bool((class_counts.min() / class_counts.max()) > 0.7)
        }
    
    # Enhanced algorithm recommendations with detailed info
    algorithm_details = {}
    for alg_name, description in available_algorithms.items():
        is_recommended = alg_name in recommended_algorithms
        confidence_score = 0.9 if is_recommended else 0.6
        
        # Algorithm-specific characteristics
        if alg_name == "random_forest":
            pros = ["Handles overfitting well", "Works with mixed data types", "Provides feature importance"]
            cons = ["Less interpretable", "Can be memory intensive"]
            training_time = "medium"
            interpretability = "medium"
        elif alg_name == "gradient_boosting":
            pros = ["High accuracy potential", "Handles complex patterns", "Feature importance"]
            cons = ["Longer training time", "Hyperparameter sensitive", "Can overfit"]
            training_time = "long"
            interpretability = "medium"
        elif alg_name == "logistic_regression":
            pros = ["Fast training", "Highly interpretable", "Good baseline"]
            cons = ["Assumes linear relationships", "Sensitive to outliers"]
            training_time = "fast"
            interpretability = "high"
        elif alg_name == "linear_regression":
            pros = ["Very fast", "Highly interpretable", "Simple baseline"]
            cons = ["Assumes linear relationships", "Sensitive to outliers"]
            training_time = "fast"
            interpretability = "high"
        else:
            pros = ["Specialized algorithm", "Good for specific cases"]
            cons = ["May require tuning"]
            training_time = "medium"
            interpretability = "medium"
        
        algorithm_details[alg_name] = {
            "name": alg_name.replace('_', ' ').title(),
            "description": description,
            "is_recommended": is_recommended,
            "confidence": confidence_score,
            "pros": pros,
            "cons": cons,
            "training_time": training_time,
            "interpretability": interpretability,
            "suitable_for": {
                "small_data": training_time == "fast",
                "large_data": alg_name in ["random_forest", "gradient_boosting"],
                "interpretability": interpretability == "high",
                "high_accuracy": alg_name in ["random_forest", "gradient_boosting"]
            }
        }
    
    detailed_data = {
        "problem_analysis": {
            "problem_type": problem_type,
            "subtype": subtype,
            "confidence": float(confidence),
            "target_variable": target_column,
            "dataset_characteristics": dataset_characteristics
        },
        "algorithm_recommendations": {
            "recommended_count": len(recommended_algorithms),
            "total_available": len(available_algorithms),
            "algorithms": algorithm_details,
            "recommended_list": recommended_algorithms
        },
        "dataset_readiness": {
            "features_ready": True,
            "target_identified": True,
            "data_quality_score": float(max(0.5, 1.0 - (dataset_characteristics["missing_values"] / (len(df) * len(df.columns))))),
            "preprocessing_complete": "feature_engineering" in PIPELINE_STATE.get("completed_operations", [])
        },
        "training_preview": {
            "estimated_time": "1-3 minutes" if dataset_characteristics["sample_size"] == "small" else "3-10 minutes",
            "memory_requirements": "low" if len(df) < 10000 else "medium",
            "cross_validation_folds": 5,
            "test_split_ratio": 0.2,
            "training_samples": int(len(df) * 0.8),
            "test_samples": int(len(df) * 0.2)
        },
        "target_analysis": {
            "column_name": str(target_column),
            "data_type": str(data_type),
            "unique_values": int(unique_values),
            "unique_ratio": float(unique_ratio),
            "sample_values": [float(x) if isinstance(x, (np.integer, np.floating)) else str(x) for x in sample_values],
            "distribution_info": dataset_characteristics.get("class_balance", {})
        }
    }

    # Store detailed data for frontend
    # PIPELINE_STATE["last_detailed_data"] = detailed_data
    
    # Before storing detailed data for frontend
    detailed_data = convert_numpy_types(detailed_data)
    PIPELINE_STATE["last_detailed_data"] = detailed_data


    return {
        "analysis_complete": True,
        "problem_type": problem_type,
        "subtype": subtype,
        "confidence": confidence,
        "target_column": target_column,
        "available_algorithms": list(available_algorithms.keys()),
        "recommended_algorithms": recommended_algorithms,
        "formatted_response": response
    }
