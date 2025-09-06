import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from helpers import DATAFRAME_CACHE, PIPELINE_STATE
from config import Config

def _get_hyperparameter_options(algorithm: str, problem_type: str) -> Dict[str, Dict[str, Any]]:
    """Get available hyperparameters for each algorithm"""
    
    hyperparameters = {}
    
    if problem_type == "classification":
        if algorithm == "random_forest":
            hyperparameters = {
                "n_estimators": {
                    "description": "Number of trees in the forest",
                    "current": 100,
                    "options": [50, 100, 200, 300, 500],
                    "type": "int"
                },
                "max_depth": {
                    "description": "Maximum depth of trees (None = unlimited)",
                    "current": None,
                    "options": [None, 5, 10, 15, 20, 30],
                    "type": "int_or_none"
                },
                "min_samples_split": {
                    "description": "Minimum samples required to split a node",
                    "current": 2,
                    "options": [2, 5, 10, 20],
                    "type": "int"
                },
                "min_samples_leaf": {
                    "description": "Minimum samples required at leaf node",
                    "current": 1,
                    "options": [1, 2, 4, 8],
                    "type": "int"
                }
            }
        elif algorithm == "logistic_regression":
            hyperparameters = {
                "C": {
                    "description": "Regularization strength (smaller = more regularization)",
                    "current": 1.0,
                    "options": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "type": "float"
                },
                "penalty": {
                    "description": "Regularization type",
                    "current": "l2",
                    "options": ["l1", "l2", "elasticnet"],
                    "type": "string"
                },
                "solver": {
                    "description": "Algorithm for optimization",
                    "current": "lbfgs",
                    "options": ["liblinear", "lbfgs", "newton-cg", "sag", "saga"],
                    "type": "string"
                }
            }
        elif algorithm == "svm":
            hyperparameters = {
                "C": {
                    "description": "Regularization parameter",
                    "current": 1.0,
                    "options": [0.1, 1.0, 10.0, 100.0],
                    "type": "float"
                },
                "kernel": {
                    "description": "Kernel type",
                    "current": "rbf",
                    "options": ["linear", "poly", "rbf", "sigmoid"],
                    "type": "string"
                },
                "gamma": {
                    "description": "Kernel coefficient (auto = 1/n_features)",
                    "current": "scale",
                    "options": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
                    "type": "string_or_float"
                }
            }
        elif algorithm == "knn":
            hyperparameters = {
                "n_neighbors": {
                    "description": "Number of neighbors",
                    "current": 5,
                    "options": [3, 5, 7, 9, 11, 15],
                    "type": "int"
                },
                "weights": {
                    "description": "Weight function",
                    "current": "uniform",
                    "options": ["uniform", "distance"],
                    "type": "string"
                },
                "metric": {
                    "description": "Distance metric",
                    "current": "minkowski",
                    "options": ["euclidean", "manhattan", "minkowski"],
                    "type": "string"
                }
            }
    
    else:  # regression
        if algorithm == "random_forest":
            hyperparameters = {
                "n_estimators": {
                    "description": "Number of trees in the forest",
                    "current": 100,
                    "options": [50, 100, 200, 300, 500],
                    "type": "int"
                },
                "max_depth": {
                    "description": "Maximum depth of trees (None = unlimited)",
                    "current": None,
                    "options": [None, 5, 10, 15, 20, 30],
                    "type": "int_or_none"
                },
                "min_samples_split": {
                    "description": "Minimum samples required to split a node",
                    "current": 2,
                    "options": [2, 5, 10, 20],
                    "type": "int"
                },
                "min_samples_leaf": {
                    "description": "Minimum samples required at leaf node",
                    "current": 1,
                    "options": [1, 2, 4, 8],
                    "type": "int"
                }
            }
        elif algorithm == "ridge_regression":
            hyperparameters = {
                "alpha": {
                    "description": "Regularization strength",
                    "current": 1.0,
                    "options": [0.1, 1.0, 10.0, 100.0, 1000.0],
                    "type": "float"
                },
                "solver": {
                    "description": "Solver algorithm",
                    "current": "auto",
                    "options": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                    "type": "string"
                }
            }
        elif algorithm == "lasso_regression":
            hyperparameters = {
                "alpha": {
                    "description": "Regularization strength",
                    "current": 1.0,
                    "options": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "type": "float"
                },
                "max_iter": {
                    "description": "Maximum iterations",
                    "current": 1000,
                    "options": [500, 1000, 2000, 5000],
                    "type": "int"
                }
            }
        elif algorithm == "svr":
            hyperparameters = {
                "C": {
                    "description": "Regularization parameter",
                    "current": 1.0,
                    "options": [0.1, 1.0, 10.0, 100.0],
                    "type": "float"
                },
                "kernel": {
                    "description": "Kernel type",
                    "current": "rbf",
                    "options": ["linear", "poly", "rbf", "sigmoid"],
                    "type": "string"
                },
                "gamma": {
                    "description": "Kernel coefficient",
                    "current": "scale",
                    "options": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
                    "type": "string_or_float"
                }
            }
        elif algorithm == "knn":
            hyperparameters = {
                "n_neighbors": {
                    "description": "Number of neighbors",
                    "current": 5,
                    "options": [3, 5, 7, 9, 11, 15],
                    "type": "int"
                },
                "weights": {
                    "description": "Weight function",
                    "current": "uniform",
                    "options": ["uniform", "distance"],
                    "type": "string"
                },
                "metric": {
                    "description": "Distance metric",
                    "current": "minkowski",
                    "options": ["euclidean", "manhattan", "minkowski"],
                    "type": "string"
                }
            }
    
    return hyperparameters

@tool
def perform_hyperparameter_tuning(path: str) -> Dict[str, Any]:
    """
    Shows available hyperparameters for the trained model and asks user for values.
    User says 'perform hyperparameter tuning' or 'tune hyperparameters'.
    """
    if path not in DATAFRAME_CACHE:
        return {"error": "Dataset not found"}

    if "models_trained" not in PIPELINE_STATE:
        response = "⚠️ **No Models for Hyperparameter Tuning**\n\n"
        response += "**You need to train a model first before tuning its hyperparameters!**\n\n"
        response += "**🚀 QUICK START:**\n"
        response += "1. **'train model'** - Detect problem type and see algorithms\n"
        response += "2. **'proceed with training [algorithm]'** - Train your chosen model\n"
        response += "3. **'perform hyperparameter tuning'** - Then optimize it!\n\n"
        
        response += "**💡 EXAMPLE:**\n"
        response += "• 'train model'\n"
        response += "• 'proceed with training random_forest'\n"
        response += "• 'perform hyperparameter tuning' ← Start here!\n\n"
        
        response += "**🎯 WHAT HYPERPARAMETER TUNING DOES:**\n"
        response += "• Optimizes algorithm settings for better performance\n"
        response += "• Shows available parameters for your trained model\n"
        response += "• Allows manual or automatic parameter optimization\n"
        
        return {
            "no_models_trained": True,
            "formatted_response": response,
            "suggestion": "Train a model first using 'train model'"
        }

    model_info = PIPELINE_STATE["models_trained"]
    
    # Get the current model (if only one trained, use that; if multiple, use best)
    if len(model_info["algorithms"]) == 1:
        current_algorithm = list(model_info["algorithms"].keys())[0]
    else:
        current_algorithm = model_info["best_algorithm"]
    
    current_score = model_info["algorithms"][current_algorithm]["score"]
    problem_type = model_info["problem_type"]
    
    # Get hyperparameter options for this algorithm
    hyperparams = _get_hyperparameter_options(current_algorithm, problem_type)
    
    if not hyperparams:
        return {
            "error": f"Hyperparameter tuning not yet supported for {current_algorithm}",
            "suggestion": "Try training a different algorithm like Random Forest or SVM"
        }
    
    # Store tuning analysis
    PIPELINE_STATE["tuning_analysis"] = {
        "current_algorithm": current_algorithm,
        "current_score": current_score,
        "problem_type": problem_type,
        "target_column": model_info["target_column"],
        "hyperparameters": hyperparams
    }
    
    # Format response
    response = "🔧 **Hyperparameter Tuning Setup**\n\n"
    response += f"**Current Model:** {current_algorithm.replace('_', ' ').title()}\n"
    response += f"**Current Performance:** {current_score:.4f}\n"
    response += f"**Problem Type:** {problem_type.title()}\n\n"
    
    response += "**⚙️ AVAILABLE HYPERPARAMETERS:**\n"
    for param_name, param_info in hyperparams.items():
        response += f"• **{param_name}** - {param_info['description']}\n"
        response += f"  Current: {param_info['current']}\n"
        response += f"  Options: {param_info['options']}\n\n"
    
    response += "**💡 NEXT STEPS:**\n"
    response += "• Type 'execute hyperparameter tuning' with your parameter choices\n"
    response += "• Format: 'execute hyperparameter tuning n_estimators=200 max_depth=10'\n"
    response += "• Or type 'auto tune hyperparameters' for automatic optimization\n\n"
    
    response += "**📝 EXAMPLES:**\n"
    if current_algorithm == "random_forest":
        response += "• 'execute hyperparameter tuning n_estimators=200 max_depth=15'\n"
        response += "• 'execute hyperparameter tuning n_estimators=300 min_samples_split=5'\n"
    elif current_algorithm == "logistic_regression":
        response += "• 'execute hyperparameter tuning C=10.0 penalty=l1'\n"
        response += "• 'execute hyperparameter tuning C=0.1 solver=liblinear'\n"
    
    return {
        "tuning_setup_complete": True,
        "current_algorithm": current_algorithm,
        "current_score": current_score,
        "available_parameters": list(hyperparams.keys()),
        "formatted_response": response
    }

@tool
def execute_hyperparameter_tuning(path: str, parameters: str = "") -> Dict[str, Any]:
    """
    Executes hyperparameter tuning with user-specified parameters.
    User says 'execute hyperparameter tuning n_estimators=200 max_depth=10'.
    """
    if path not in DATAFRAME_CACHE:
        return {"error": "Dataset not found"}

    if "tuning_analysis" not in PIPELINE_STATE:
        return {"error": "Please run hyperparameter setup first using 'perform hyperparameter tuning'"}

    tuning_info = PIPELINE_STATE["tuning_analysis"]
    algorithm = tuning_info["current_algorithm"]
    problem_type = tuning_info["problem_type"]
    target_column = tuning_info["target_column"]
    old_score = tuning_info["current_score"]
    
    df = DATAFRAME_CACHE[path]
    
    # Parse user parameters
    user_params = {}
    if parameters:
        try:
            # Parse parameters like "n_estimators=200 max_depth=10"
            param_pairs = parameters.split()
            for pair in param_pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert value to appropriate type
                    if value.lower() == 'none':
                        user_params[key] = None
                    elif value.lower() in ['true', 'false']:
                        user_params[key] = value.lower() == 'true'
                    elif value.replace('.', '').replace('-', '').isdigit():
                        user_params[key] = float(value) if '.' in value else int(value)
                    else:
                        user_params[key] = value
        except Exception as e:
            return {"error": f"Error parsing parameters: {str(e)}. Use format: 'param1=value1 param2=value2'"}
    
    if not user_params:
        return {"error": "No parameters provided. Use format: 'n_estimators=200 max_depth=10'"}
    
    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    
    # Create model with new hyperparameters
    try:
        if problem_type == "classification":
            if algorithm == "random_forest":
                model = RandomForestClassifier(random_state=Config.RANDOM_STATE, **user_params)
            elif algorithm == "logistic_regression":
                model = LogisticRegression(random_state=Config.RANDOM_STATE, max_iter=Config.MAX_ITER, **user_params)
            elif algorithm == "svm":
                model = SVC(random_state=Config.RANDOM_STATE, **user_params)
            elif algorithm == "knn":
                model = KNeighborsClassifier(**user_params)
            else:
                return {"error": f"Hyperparameter tuning not supported for {algorithm}"}
        else:  # regression
            if algorithm == "random_forest":
                model = RandomForestRegressor(random_state=Config.RANDOM_STATE, **user_params)
            elif algorithm == "ridge_regression":
                model = Ridge(random_state=Config.RANDOM_STATE, **user_params)
            elif algorithm == "lasso_regression":
                model = Lasso(random_state=Config.RANDOM_STATE, **user_params)
            elif algorithm == "svr":
                model = SVR(**user_params)
            elif algorithm == "knn":
                model = KNeighborsRegressor(**user_params)
            else:
                return {"error": f"Hyperparameter tuning not supported for {algorithm}"}
        
        # Train model with new parameters
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate new score
        if problem_type == "classification":
            new_score = accuracy_score(y_test, y_pred)
            metric_name = "accuracy"
        else:
            new_score = r2_score(y_test, y_pred)
            metric_name = "r2_score"
        
        # Calculate improvement
        improvement = new_score - old_score
        improvement_pct = (improvement / old_score) * 100 if old_score > 0 else 0
        
        # Update model in pipeline state
        PIPELINE_STATE["models_trained"]["algorithms"][algorithm] = {
            "model": model,
            "score": new_score,
            "predictions": y_pred,
            "hyperparameters": user_params
        }
        PIPELINE_STATE["models_trained"]["best_score"] = new_score
        
    except Exception as e:
        return {"error": f"Error training model with new parameters: {str(e)}"}
    
    # Format response
    response = "✅ **Hyperparameter Tuning Complete!**\n\n"
    response += f"**Algorithm:** {algorithm.replace('_', ' ').title()}\n"
    response += f"**Parameters Used:** {user_params}\n\n"
    
    response += "**📊 PERFORMANCE COMPARISON:**\n"
    response += f"• **Old {metric_name}:** {old_score:.4f}\n"
    response += f"• **New {metric_name}:** {new_score:.4f}\n"
    
    if improvement > 0:
        response += f"• **Improvement:** +{improvement:.4f} ({improvement_pct:+.1f}%) 🎉\n"
        status = "IMPROVED ✅"
    elif improvement < 0:
        response += f"• **Change:** {improvement:.4f} ({improvement_pct:+.1f}%) 📉\n"
        status = "DECREASED ⚠️"
    else:
        response += f"• **Change:** No change\n"
        status = "SAME ➡️"
    
    response += f"\n**🎯 RESULT:** Performance {status}\n\n"
    
    if improvement > 0:
        response += "**💡 SUCCESS! Your hyperparameter tuning improved the model.**\n"
    else:
        response += "**💡 TIP:** Try different parameter values or consider other improvement techniques.\n"
    
    response += "\n**🔄 NEXT STEPS:**\n"
    response += "• Type 'evaluate model' for detailed analysis\n"
    response += "• Type 'perform hyperparameter tuning' to try different parameters\n"
    response += "• Type 'improve model' for other improvement suggestions\n"
    
    return {
        "tuning_complete": True,
        "algorithm": algorithm,
        "old_score": old_score,
        "new_score": new_score,
        "improvement": improvement,
        "improvement_percentage": improvement_pct,
        "parameters_used": user_params,
        "formatted_response": response,
        "status": status
    }

@tool
def auto_tune_hyperparameters(path: str) -> Dict[str, Any]:
    """
    Automatically finds best hyperparameters using GridSearchCV.
    User says 'auto tune hyperparameters' for automatic optimization.
    """
    if path not in DATAFRAME_CACHE:
        return {"error": "Dataset not found"}

    if "tuning_analysis" not in PIPELINE_STATE:
        return {"error": "Please run hyperparameter setup first using 'perform hyperparameter tuning'"}

    tuning_info = PIPELINE_STATE["tuning_analysis"]
    algorithm = tuning_info["current_algorithm"]
    problem_type = tuning_info["problem_type"]
    target_column = tuning_info["target_column"]
    old_score = tuning_info["current_score"]
    
    df = DATAFRAME_CACHE[path]
    
    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    
    # Define parameter grids for automatic tuning
    param_grids = {}
    
    if problem_type == "classification":
        if algorithm == "random_forest":
            base_model = RandomForestClassifier(random_state=Config.RANDOM_STATE)
            param_grids = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif algorithm == "logistic_regression":
            base_model = LogisticRegression(random_state=Config.RANDOM_STATE, max_iter=Config.MAX_ITER)
            param_grids = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs']
            }
    else:  # regression
        if algorithm == "random_forest":
            base_model = RandomForestRegressor(random_state=Config.RANDOM_STATE)
            param_grids = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif algorithm == "ridge_regression":
            base_model = Ridge(random_state=Config.RANDOM_STATE)
            param_grids = {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
    
    if not param_grids:
        return {"error": f"Auto-tuning not yet supported for {algorithm}"}
    
    try:
        # Perform grid search
        scoring = 'accuracy' if problem_type == "classification" else 'r2'
        grid_search = GridSearchCV(
            base_model, 
            param_grids, 
            cv=3, 
            scoring=scoring,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model and score
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        if problem_type == "classification":
            new_score = accuracy_score(y_test, y_pred)
            metric_name = "accuracy"
        else:
            new_score = r2_score(y_test, y_pred)
            metric_name = "r2_score"
        
        best_params = grid_search.best_params_
        improvement = new_score - old_score
        improvement_pct = (improvement / old_score) * 100 if old_score > 0 else 0
        
        # Update model in pipeline state
        PIPELINE_STATE["models_trained"]["algorithms"][algorithm] = {
            "model": best_model,
            "score": new_score,
            "predictions": y_pred,
            "hyperparameters": best_params
        }
        PIPELINE_STATE["models_trained"]["best_score"] = new_score
        
    except Exception as e:
        return {"error": f"Error during automatic tuning: {str(e)}"}
    
    # Format response
    response = "🤖 **Automatic Hyperparameter Tuning Complete!**\n\n"
    response += f"**Algorithm:** {algorithm.replace('_', ' ').title()}\n"
    response += f"**Search Method:** Grid Search with 3-fold Cross-Validation\n\n"
    
    response += "**🏆 BEST PARAMETERS FOUND:**\n"
    for param, value in best_params.items():
        response += f"• **{param}:** {value}\n"
    
    response += "\n**📊 PERFORMANCE COMPARISON:**\n"
    response += f"• **Old {metric_name}:** {old_score:.4f}\n"
    response += f"• **New {metric_name}:** {new_score:.4f}\n"
    
    if improvement > 0:
        response += f"• **Improvement:** +{improvement:.4f} ({improvement_pct:+.1f}%) 🎉\n"
        status = "IMPROVED ✅"
    elif improvement < 0:
        response += f"• **Change:** {improvement:.4f} ({improvement_pct:+.1f}%) 📉\n"
        status = "DECREASED ⚠️"
    else:
        response += f"• **Change:** No change\n"
        status = "SAME ➡️"
    
    response += f"\n**🎯 RESULT:** Automatic tuning {status}\n\n"
    
    if improvement > 0:
        response += "**💡 SUCCESS! Automatic tuning found better parameters.**\n"
    else:
        response += "**💡 INFO:** Current parameters were already near-optimal.\n"
    
    response += "\n**🔄 NEXT STEPS:**\n"
    response += "• Type 'evaluate model' for detailed analysis\n"
    response += "• Type 'improve model' for other improvement techniques\n"
    
    return {
        "auto_tuning_complete": True,
        "algorithm": algorithm,
        "old_score": old_score,
        "new_score": new_score,
        "improvement": improvement,
        "improvement_percentage": improvement_pct,
        "best_parameters": best_params,
        "formatted_response": response,
        "status": status
    }