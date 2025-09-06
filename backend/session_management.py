from typing import Dict, Any
from langchain_core.tools import tool
from helpers import DATAFRAME_CACHE, PIPELINE_STATE

@tool
def end_session(confirmation: str = "no") -> Dict[str, Any]:
    """
    Ends current ML pipeline session and clears all data.
    User says 'end session', 'reset pipeline', or 'start fresh'.
    """
    if confirmation.lower() not in ["yes", "confirm", "y"]:
        response = "⚠️ **End Session Confirmation**\n\n"
        response += "**This will permanently delete:**\n"
        response += "• All loaded datasets\n"
        response += "• Trained models and results\n"
        response += "• Pipeline progress and analysis\n"
        response += "• Cleaning and feature engineering work\n\n"
        response += "**💡 To confirm, type:**\n"
        response += "• 'end session yes'\n"
        response += "• 'reset pipeline confirm'\n"
        response += "• 'start fresh y'\n\n"
        response += "**Or continue working with current session.**"
        
        return {
            "confirmation_required": True,
            "formatted_response": response,
            "current_datasets": list(DATAFRAME_CACHE.keys()),
            "pipeline_operations": PIPELINE_STATE.get("completed_operations", [])
        }
    
    # Clear all data
    datasets_cleared = list(DATAFRAME_CACHE.keys())
    operations_cleared = PIPELINE_STATE.get("completed_operations", [])
    
    DATAFRAME_CACHE.clear()
    PIPELINE_STATE.clear()
    
    # Reset pipeline state to initial values
    PIPELINE_STATE.update({
        "current_stage": None,
        "completed_operations": [],
        "dataset_info": {},
        "cleaning_analysis": {},
        "feature_analysis": {},
        "models_trained": {},
        "evaluation_results": {}
    })
    
    response = "✅ **Session Ended Successfully**\n\n"
    response += "**🗑️ CLEARED DATA:**\n"
    if datasets_cleared:
        response += f"• **Datasets:** {', '.join(datasets_cleared)}\n"
    if operations_cleared:
        response += f"• **Operations:** {', '.join(operations_cleared)}\n"
    response += "• **Models:** All trained models removed\n"
    response += "• **Analysis:** All pipeline analysis cleared\n\n"
    
    response += "**🆕 FRESH START:**\n"
    response += "• Ready to load new dataset\n"
    response += "• All pipeline stages reset\n"
    response += "• Memory cleared for optimal performance\n\n"
    
    response += "**💡 NEXT STEPS:**\n"
    response += "• Load a new dataset to begin ML pipeline\n"
    response += "• Type 'help' to see available commands\n"
    response += "• Ask general ML questions anytime\n"
    response += "• Type 'exit' or 'quit' to leave the application\n"
    
    return {
        "session_ended": True,
        "datasets_cleared": datasets_cleared,
        "operations_cleared": operations_cleared,
        "formatted_response": response,
        "ready_for_new_session": True
    }

@tool
def show_session_status() -> Dict[str, Any]:
    """
    Shows current session status and progress.
    User says 'session status', 'show progress', or 'what have we done'.
    """
    response = "📊 **Current Session Status**\n\n"
    
    # Dataset information
    if DATAFRAME_CACHE:
        response += "**📁 LOADED DATASETS:**\n"
        for dataset_name in DATAFRAME_CACHE.keys():
            df = DATAFRAME_CACHE[dataset_name]
            response += f"• **{dataset_name}** - {df.shape[0]} rows × {df.shape[1]} columns\n"
    else:
        response += "**📁 DATASETS:** None loaded\n"
    
    response += "\n"
    
    # Pipeline progress
    completed_ops = PIPELINE_STATE.get("completed_operations", [])
    if completed_ops:
        response += "**✅ COMPLETED OPERATIONS:**\n"
        for op in completed_ops:
            response += f"• {op.replace('_', ' ').title()}\n"
    else:
        response += "**✅ COMPLETED OPERATIONS:** None\n"
    
    response += "\n"
    
    # Current analysis status
    analysis_status = []
    if "cleaning_analysis" in PIPELINE_STATE:
        analysis_status.append("Data Cleaning Analysis")
    if "feature_analysis" in PIPELINE_STATE:
        analysis_status.append("Feature Engineering Analysis")
    if "training_analysis" in PIPELINE_STATE:
        analysis_status.append("Problem Type Detection")
    if "models_trained" in PIPELINE_STATE:
        models_count = len(PIPELINE_STATE["models_trained"].get("algorithms", {}))
        analysis_status.append(f"Model Training ({models_count} models)")
    if "clustering_results" in PIPELINE_STATE:
        analysis_status.append("Clustering Analysis")
    
    if analysis_status:
        response += "**🔍 AVAILABLE ANALYSIS:**\n"
        for status in analysis_status:
            response += f"• {status}\n"
    else:
        response += "**🔍 AVAILABLE ANALYSIS:** None\n"
    
    response += "\n**💡 ACTIONS:**\n"
    response += "• Type 'help' to see available commands\n"
    response += "• Type 'end session' to start fresh\n"
    response += "• Continue with current pipeline\n"
    
    return {
        "status_shown": True,
        "datasets_loaded": len(DATAFRAME_CACHE),
        "operations_completed": len(completed_ops),
        "analysis_available": len(analysis_status),
        "formatted_response": response,
        "session_active": len(DATAFRAME_CACHE) > 0 or len(completed_ops) > 0
    }