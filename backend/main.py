import pandas as pd
import os
from pathlib import Path
from agent_setup import create_agent_executor
from helpers import DATAFRAME_CACHE

def get_dataset_path():
    """Get dataset path from user input"""
    print("📁 **Dataset Selection**")
    print("Options:")
    print("1. Enter full path to your CSV file")
    print("2. Place CSV in 'data' folder and enter filename")
    
    # Create data folder if it doesn't exist
    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)
    
    while True:
        user_input = input("\n📂 Enter dataset path or filename: ").strip()
        
        # Check if it's a full path
        if os.path.isfile(user_input):
            return user_input
        
        # Check if it's in data folder
        data_path = data_folder / user_input
        if data_path.exists():
            return str(data_path)
        
        # Try common extensions
        for ext in ['.csv', '.CSV']:
            test_path = data_folder / f"{user_input}{ext}"
            if test_path.exists():
                return str(test_path)
        
        print(f"❌ File not found: {user_input}")
        print(f"💡 Make sure to place your CSV file in the 'data' folder or provide full path")

def main():
    """Main execution function for local environment"""
    
    print("🤖 **Local ML Assistant Starting...**")
    
    # Get dataset path
    dataset_path = get_dataset_path()
    
    try:
        # Load dataset
        df = pd.read_csv(dataset_path)
        filename = os.path.basename(dataset_path)
        
        # Cache with both full path and filename for flexibility
        DATAFRAME_CACHE[dataset_path] = df
        DATAFRAME_CACHE[filename] = df
        
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"📁 File: {filename}")

        # Create agent
        agent_executor = create_agent_executor(filename)

        print("\n🤖 **Natural ML Assistant Ready!**")
        print("\n� **Typical ML Workflow:** Explore → Clean → Feature Engineering → Training → Evaluation")
        print("\n⚡ **Quick Start:** Try 'explore my data' to begin your ML journey!")
        print("\n💬 **Just tell me what you want to do:**")
        print("• 'explore my data'")
        print("• 'clean the dataset'")
        print("• 'engineer features'")
        print("• 'train a model'")
        print("• 'evaluate performance'")
        print("\n🔥 No need to know technical commands - just speak naturally!")
        print("💡 Type 'help' for more options or 'exit' to quit")

        # Interactive loop
        while True:
            user_input = input("\n💬 You: ")
            
            if user_input.strip().lower() in ['exit', 'quit', 'bye']:
                print("👋 Thanks for using the ML Assistant!")
                break
            elif user_input.strip().lower() == 'help':
                print("\n📋 **Available Commands:**")
                print("• 'explore data' - Get dataset overview")
                print("• 'perform cleaning' - Analyze data quality")
                print("• 'proceed with cleaning' - Execute cleaning")
                print("• 'perform feature engineering' - Analyze features")
                print("• 'proceed with feature engineering' - Transform features")
                print("• 'detect problem type' - Identify classification/regression/clustering")
                print("• 'show algorithms' - See available models for your problem")
                print("• 'train model' - Detect problem type and show available algorithms")
                print("• 'proceed with training [algorithm]' - Train specific model")
                print("• 'proceed with training recommended' - Train best algorithms")
                print("• 'perform clustering' - Unsupervised clustering analysis")
                print("• 'evaluate model' - Check performance")
                print("• 'compare models' - Compare all trained models")
                print("• 'improve model' - Get improvement suggestions")
                print("• 'perform hyperparameter tuning' - Setup parameter optimization")
                print("• 'execute hyperparameter tuning [params]' - Apply specific parameters")
                print("• 'auto tune hyperparameters' - Automatic optimization")
                print("• 'end session' - Clear all data and start fresh")
                print("• 'session status' - Show current progress")
                print("• Ask any ML question (e.g., 'what is hyperparameter tuning?')")
                continue

            try:
                result = agent_executor.invoke({
                    "input": user_input,
                    "uploaded_filename": filename
                })
                print(f"\n🤖 **Assistant:** {result['output']}")

            except Exception as e:
                print(f"❌ Sorry, I encountered an error: {str(e)}")
                print("💡 Try rephrasing your request or type 'help' for available commands")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 Make sure your CSV file is properly formatted and accessible")

if __name__ == "__main__":
    main()
