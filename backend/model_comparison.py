from typing import Dict, Any
from langchain_core.tools import tool
from helpers import PIPELINE_STATE
import pandas as pd

@tool
def compare_models(path: str) -> Dict[str, Any]:
    """
    Provides detailed comparison of all trained models.
    User says 'compare models' or 'show model comparison'.
    """
    if "models_trained" not in PIPELINE_STATE:
        response = "⚠️ **No Models to Compare**\n\n"
        response += "**You need to train multiple models first!**\n\n"
        response += "**🚀 HOW TO GET MULTIPLE MODELS:**\n"
        response += "1. **'train model'** - Detect problem type\n"
        response += "2. **'proceed with training recommended'** - Trains multiple algorithms\n"
        response += "3. **'compare models'** - Then compare their performance!\n\n"
        
        response += "**💡 ALTERNATIVE:**\n"
        response += "• Train individual models and compare:\n"
        response += "  - 'proceed with training random_forest'\n"
        response += "  - 'proceed with training gradient_boosting'\n"
        response += "  - 'compare models'\n\n"
        
        response += "**❓ Need help?** Type 'help' for all available commands"
        
        return {
            "no_models_trained": True,
            "formatted_response": response,
            "suggestion": "Train multiple models first using 'proceed with training recommended'"
        }

    model_info = PIPELINE_STATE["models_trained"]
    algorithms = model_info["algorithms"]
    
    # Create comparison table
    comparison_data = []
    for alg_name, results in algorithms.items():
        comparison_data.append({
            "Algorithm": alg_name.replace('_', ' ').title(),
            "Score": f"{results['score']:.4f}",
            "Rank": ""
        })
    
    # Sort by score and add ranks
    comparison_data.sort(key=lambda x: float(x["Score"]), reverse=True)
    for i, item in enumerate(comparison_data):
        item["Rank"] = f"#{i+1}"
        if i == 0:
            item["Status"] = "🏆 BEST"
        elif i == 1:
            item["Status"] = "🥈 2nd"
        elif i == 2:
            item["Status"] = "🥉 3rd"
        else:
            item["Status"] = ""
    
    # Format response
    response = "🏆 **Model Performance Comparison**\n\n"
    response += f"**Problem Type:** {model_info['problem_type'].title()}\n"
    response += f"**Target Variable:** {model_info['target_column']}\n"
    response += f"**Models Trained:** {len(algorithms)}\n\n"
    
    metric = "Accuracy" if model_info["problem_type"] == "classification" else "R² Score"
    response += f"**📊 RANKING BY {metric.upper()}:**\n"
    
    for item in comparison_data:
        response += f"{item['Rank']} **{item['Algorithm']}** - {item['Score']} {item.get('Status', '')}\n"
    
    # Performance insights
    best_score = float(comparison_data[0]["Score"])
    worst_score = float(comparison_data[-1]["Score"])
    score_diff = best_score - worst_score
    
    response += f"\n**📈 INSIGHTS:**\n"
    response += f"• Best performing: {comparison_data[0]['Algorithm']} ({comparison_data[0]['Score']})\n"
    response += f"• Performance gap: {score_diff:.4f} between best and worst\n"
    
    if score_diff < 0.05:
        response += "• Models perform similarly - consider ensemble methods\n"
    elif best_score > 0.9:
        response += "• Excellent performance achieved!\n"
    elif best_score < 0.7:
        response += "• Consider feature engineering or hyperparameter tuning\n"
    
    return {
        "comparison_complete": True,
        "models_compared": len(algorithms),
        "best_model": model_info["best_algorithm"],
        "formatted_response": response,
        "comparison_data": comparison_data
    }

@tool
def suggest_improvements(path: str) -> Dict[str, Any]:
    """
    Suggests improvements based on model performance.
    User says 'improve model' or 'suggest improvements'.
    """
    if "models_trained" not in PIPELINE_STATE:
        response = "⚠️ **No Models to Improve**\n\n"
        response += "**You need to train a model first before improving it!**\n\n"
        response += "**🚀 QUICK START:**\n"
        response += "1. **'train model'** - Detect problem type and see algorithms\n"
        response += "2. **'proceed with training [algorithm]'** - Train your model\n"
        response += "3. **'improve model'** - Get improvement suggestions!\n\n"
        
        response += "**💡 EXAMPLE WORKFLOW:**\n"
        response += "• 'train model'\n"
        response += "• 'proceed with training random_forest'\n"
        response += "• 'improve model' ← You'll be here!\n\n"
        
        response += "**🎯 WHAT YOU'LL GET:**\n"
        response += "• Performance-based improvement suggestions\n"
        response += "• Hyperparameter tuning recommendations\n"
        response += "• Alternative algorithm suggestions\n"
        
        return {
            "no_models_trained": True,
            "formatted_response": response,
            "suggestion": "Train a model first using 'train model'"
        }

    model_info = PIPELINE_STATE["models_trained"]
    best_score = model_info["best_score"]
    problem_type = model_info["problem_type"]
    
    response = "💡 **Model Improvement Suggestions**\n\n"
    response += f"**Current Best Score:** {best_score:.4f}\n\n"
    
    suggestions = []
    
    # Get current trained algorithm for specific suggestions
    current_algorithm = model_info.get("best_algorithm", "unknown")
    
    # Performance-based suggestions
    if problem_type == "classification":
        if best_score < 0.7:
            suggestions.extend([
                "🔧 **Try Different Algorithm**: 'train model' → try Random Forest or Gradient Boosting",
                "⚙️ **Hyperparameter Tuning**: 'perform hyperparameter tuning' to optimize current model",
                "📊 **Data Quality**: Check for class imbalance and apply SMOTE if needed",
                "🎯 **Feature Engineering**: Create polynomial features or interaction terms"
            ])
        elif best_score < 0.85:
            suggestions.extend([
                "⚙️ **Hyperparameter Tuning**: 'perform hyperparameter tuning' for better performance",
                "🔧 **Try Different Algorithm**: 'train model' → experiment with SVM or Gradient Boosting",
                "🔄 **Cross-Validation**: Use k-fold CV for better evaluation",
                "🎯 **Ensemble**: Combine multiple models with voting classifier"
            ])
        else:
            suggestions.extend([
                "✅ **Great Performance!** Consider:",
                "⚙️ **Fine-tuning**: 'perform hyperparameter tuning' for marginal improvements",
                "🔄 **Validation**: Test on additional datasets",
                "📈 **Production**: Ready for deployment!"
            ])
    else:  # regression
        if best_score < 0.6:
            suggestions.extend([
                "🔧 **Try Different Algorithm**: 'train model' → try Random Forest or Gradient Boosting",
                "⚙️ **Hyperparameter Tuning**: 'perform hyperparameter tuning' to optimize current model",
                "📊 **Feature Engineering**: Add polynomial features, log transforms",
                "🎯 **Outlier Treatment**: Review outlier handling strategy"
            ])
        elif best_score < 0.8:
            suggestions.extend([
                "⚙️ **Hyperparameter Tuning**: 'perform hyperparameter tuning' for better performance",
                "🔧 **Try Different Algorithm**: 'train model' → experiment with SVR or ensemble methods",
                "🔄 **Feature Selection**: Use recursive feature elimination",
                "🎯 **Regularization**: Experiment with Ridge/Lasso parameters"
            ])
        else:
            suggestions.extend([
                "🌟 **Excellent Performance!** Consider:",
                "⚙️ **Fine-tuning**: 'perform hyperparameter tuning' for marginal improvements",
                "🔄 **Robustness**: Test with cross-validation",
                "📈 **Production**: Ready for deployment!"
            ])
    
    # Add specific action suggestions
    suggestions.extend([
        "📋 **Available Actions:**",
        "• 'perform hyperparameter tuning' - Optimize current model parameters",
        "• 'train model' → 'proceed with training [algorithm]' - Try different algorithm",
        "• 'compare models' - See detailed model comparison",
        "• 'evaluate model' - Get comprehensive performance metrics"
    ])
    
    # Add algorithm-specific hyperparameter suggestions
    if current_algorithm in ["random_forest", "logistic_regression", "svm", "ridge_regression", "lasso_regression"]:
        suggestions.append(f"💡 **{current_algorithm.replace('_', ' ').title()}** supports hyperparameter tuning!")
    
    for suggestion in suggestions:
        response += f"{suggestion}\n"
    
    return {
        "suggestions_provided": True,
        "current_score": best_score,
        "formatted_response": response,
        "improvement_areas": len([s for s in suggestions if s.startswith(("🔧", "📊", "🎯", "⚙️"))])
    }