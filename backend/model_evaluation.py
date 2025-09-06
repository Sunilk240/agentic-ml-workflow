from typing import Dict, Any
from langchain_core.tools import tool
from helpers import PIPELINE_STATE

@tool
def evaluate_model(path: str) -> Dict[str, Any]:
    """
    Provides detailed evaluation of trained models.
    User says 'evaluate model' or 'show model performance'.
    """
    if "models_trained" not in PIPELINE_STATE:
        response = "⚠️ **No Models Trained Yet**\n\n"
        response += "**To evaluate models, you need to train them first!**\n\n"
        response += "**🚀 QUICK START:**\n"
        response += "1. **'train model'** - Detect problem type and see available algorithms\n"
        response += "2. **'proceed with training [algorithm]'** - Train your chosen model\n"
        response += "3. **'evaluate model'** - Then come back here for evaluation!\n\n"
        
        response += "**💡 EXAMPLES:**\n"
        response += "• 'train model' → 'proceed with training random_forest'\n"
        response += "• 'train model' → 'proceed with training recommended'\n\n"
        
        response += "**❓ Need help?** Type 'help' for all available commands"
        
        return {
            "no_models_trained": True,
            "formatted_response": response,
            "suggestion": "Train models first using 'train model'"
        }

    model_info = PIPELINE_STATE["models_trained"]
    
    # Additional safety check for problem_type
    if "problem_type" not in model_info:
        response = "⚠️ **Incomplete Model Information**\n\n"
        response += "**There seems to be an issue with the trained model data.**\n\n"
        response += "**🔄 SUGGESTED SOLUTION:**\n"
        response += "• Try training your model again using 'train model'\n"
        response += "• This will ensure all model information is properly stored\n\n"
        response += "**💡 If the issue persists:**\n"
        response += "• Type 'session status' to check current progress\n"
        response += "• Type 'end session yes' to start fresh\n"
        
        return {
            "incomplete_model_info": True,
            "formatted_response": response,
            "suggestion": "Retrain models to fix missing information"
        }

    response = "📊 **Detailed Model Evaluation Report**\n\n"
    response += f"**Problem Type:** {model_info['problem_type'].title()}\n"
    response += f"**Target Variable:** {model_info['target_column']}\n\n"

    response += "**🔍 ALGORITHM COMPARISON:**\n"

    for alg_name, results in model_info["algorithms"].items():
        score = results["score"]
        metric = "accuracy" if model_info["problem_type"] == "classification" else "R² score"

        # Performance interpretation
        if model_info["problem_type"] == "classification":
            if score >= 0.90:
                performance = "Excellent 🌟"
            elif score >= 0.80:
                performance = "Very Good 👍"
            elif score >= 0.70:
                performance = "Good ✓"
            elif score >= 0.60:
                performance = "Fair ⚠️"
            else:
                performance = "Needs Improvement 🔧"
        else:  # regression
            if score >= 0.90:
                performance = "Excellent 🌟"
            elif score >= 0.80:
                performance = "Very Good 👍"
            elif score >= 0.60:
                performance = "Good ✓"
            elif score >= 0.40:
                performance = "Fair ⚠️"
            else:
                performance = "Needs Improvement 🔧"

        response += f"• **{alg_name.replace('_', ' ').title()}**\n"
        response += f"  - {metric}: {score:.4f}\n"
        response += f"  - Performance: {performance}\n\n"

    best_alg = model_info["best_algorithm"]
    response += f"**🏆 BEST MODEL:** {best_alg.replace('_', ' ').title()}\n\n"

    # Recommendations based on performance
    best_score = model_info["best_score"]
    response += "**🎯 RECOMMENDATIONS:**\n"

    if model_info["problem_type"] == "classification":
        if best_score < 0.70:
            response += "• Consider feature engineering to improve model performance\n"
            response += "• Try ensemble methods or gradient boosting algorithms\n"
            response += "• Check for class imbalance in your target variable\n"
        elif best_score < 0.85:
            response += "• Fine-tune hyperparameters for better performance\n"
            response += "• Consider cross-validation for more robust evaluation\n"
        else:
            response += "• Excellent performance! Model is ready for deployment\n"
            response += "• Consider testing on additional validation data\n"
    else:  # regression
        if best_score < 0.60:
            response += "• Add more relevant features or engineer new ones\n"
            response += "• Try polynomial features or interaction terms\n"
            response += "• Consider ensemble methods\n"
        elif best_score < 0.80:
            response += "• Fine-tune model hyperparameters\n"
            response += "• Validate with cross-validation\n"
        else:
            response += "• Great performance! Model captures data patterns well\n"
            response += "• Ready for production deployment\n"

    return {
        "evaluation_complete": True,
        "best_model": best_alg,
        "best_score": best_score,
        "formatted_response": response,
        "performance_level": performance
    }
