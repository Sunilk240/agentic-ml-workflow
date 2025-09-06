from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from config import Config
from data_exploration import explore_dataset
from data_cleaning import perform_data_cleaning, execute_data_cleaning
from feature_engineering import perform_feature_engineering, execute_feature_engineering
from model_training import train_model, execute_model_training
from model_evaluation import evaluate_model
from model_comparison import compare_models, suggest_improvements
from model_improvement import perform_hyperparameter_tuning, execute_hyperparameter_tuning, auto_tune_hyperparameters
from problem_detection import detect_problem_type, show_available_algorithms
from clustering import perform_clustering, execute_clustering, analyze_clusters
from session_management import end_session, show_session_status
from general_assistant import answer_ml_question, explain_pipeline_concept

def create_natural_agent_gemini():
    """Creates an agent with natural, high-level tools for Gemini."""
    
    # Only user-facing tools
    user_tools = [
        explore_dataset,
        perform_data_cleaning,
        execute_data_cleaning,
        perform_feature_engineering,
        execute_feature_engineering,
        detect_problem_type,
        show_available_algorithms,
        train_model,
        execute_model_training,
        # evaluate_model,
        # compare_models,
        # suggest_improvements,
        perform_hyperparameter_tuning,
        execute_hyperparameter_tuning,
        auto_tune_hyperparameters,
        perform_clustering,
        execute_clustering,
        analyze_clusters,
        end_session,
        show_session_status,
        answer_ml_question,
        explain_pipeline_concept
    ]

    # OLD PROMPT - COMMENTED OUT
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system",
    #      "You are an ML assistant. Use the available tools to perform tasks directly.\n\n"
    #      "TOOL MAPPING:\n"
    #      "• 'explore data' → explore_dataset\n"
    #      "• 'perform cleaning' → perform_data_cleaning\n"
    #      "• 'proceed with cleaning' → execute_data_cleaning\n"
    #      "• 'perform feature engineering' → perform_feature_engineering\n"
    #      "• 'proceed with feature engineering' → execute_feature_engineering\n"
    #      "• 'train model' → train_model\n"
    #      "• 'proceed with training' → execute_model_training\n"
    #      "• 'evaluate model' → evaluate_model\n"
    #      "• 'compare models' → compare_models\n"
    #      "• 'end session' → end_session\n\n"
    #      "RULES:\n"
    #      "- When user says 'train model', call train_model with path='{uploaded_filename}'\n"
    #      "- Execute tools directly, don't ask clarifying questions\n"
    #      "- Use the dataset: {uploaded_filename}"),
    #     ("user", "{input}"),
    #     ("placeholder", "{agent_scratchpad}")
    # ])

    # NEW CONTEXT-AWARE PROMPT
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an ML pipeline assistant that maintains context throughout the workflow. "
         "Always use the available tools and maintain pipeline state.\n\n"
         "WORKFLOW STAGES & TOOL MAPPING:\n"
         "1. EXPLORATION:\n"
         "   • 'explore data' → explore_dataset(path='{uploaded_filename}')\n\n"
         "2. DATA CLEANING:\n"
         "   • 'perform cleaning' → perform_data_cleaning(path='{uploaded_filename}')\n"
         "   • 'proceed with cleaning' OR 'proceed with recommended' → execute_data_cleaning(path='{uploaded_filename}')\n\n"
         "3. FEATURE ENGINEERING:\n"
         "   • 'perform feature engineering' → perform_feature_engineering(path='{uploaded_filename}')\n"
         "   • 'proceed with feature engineering' → execute_feature_engineering(path='{uploaded_filename}')\n\n"
         "4. MODEL TRAINING:\n"
         "   • 'train model' → train_model(path='{uploaded_filename}')\n"
         "   • 'proceed with training [algorithm]' → execute_model_training(path='{uploaded_filename}', algorithm='[algorithm]')\n"
         "   • 'proceed with training recommended' → execute_model_training(path='{uploaded_filename}', algorithm='recommended')\n\n"
         "4b. CLUSTERING:\n"
         "   • 'perform clustering' → perform_clustering(path='{uploaded_filename}') - shows available algorithms\n"
         "   • 'execute clustering [algorithm]' → execute_clustering(path='{uploaded_filename}', algorithm='[algorithm]')\n"
         "   • 'execute clustering recommended' → execute_clustering(path='{uploaded_filename}', algorithm='recommended')\n\n"
         "CRITICAL RULES:\n"
         "- ALWAYS use path='{uploaded_filename}' for all tool calls\n"
         "- Maintain pipeline context - don't ask for information already available in pipeline state\n"
         "- For 'evaluate' without specifying model, evaluate the most recently trained model\n"
         "- For cleaning/feature engineering, show alternatives and let user choose method\n"
         "- For training, show all available algorithms and let user choose\n"
         "- Execute tools directly based on user intent, don't ask clarifying questions\n"
         "- Pipeline state is maintained automatically - use it to provide context-aware responses\n"
         "- If user asks about 'deploy' or 'deployment', respond: 'We do not perform model deployment. This pipeline focuses on model development, training, and evaluation.'\n"
         "- For clustering: Always show available algorithms (K-Means, Hierarchical, DBSCAN, Gaussian Mixture) and ask user to choose before executing\n"
         "- After 'end session' completes: Always suggest 'Type exit or quit to leave the application' instead of suggesting new dataset loading\n\n"
         "- If user do no give number of clusters in case of clustering use default number of clusters equal to 3 \n"
         "Current dataset: {uploaded_filename}"),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    return user_tools, prompt

def initialize_llm_gemini():
    """Initialize the Gemini LLM with configuration"""
    Config.validate_config()
    
    # Use Gemini model
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=Config.GEMINI_API_KEY,  # You'll need to add this to config
        temperature=0.1,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

def create_agent_executor_gemini(dataset_path: str):
    """Create and return the Gemini agent executor"""
    llm = initialize_llm_gemini()
    tools, prompt_template = create_natural_agent_gemini()
    
    # Load dataset into cache immediately
    from helpers import DATAFRAME_CACHE
    import pandas as pd
    import os
    
    try:
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            DATAFRAME_CACHE[dataset_path] = df
            print(f"✅ Dataset loaded into cache: {dataset_path}")
            print(f"📊 Dataset shape: {df.shape}")
        else:
            print(f"❌ Dataset file not found: {dataset_path}")
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
    
    # Debug: Print tool names and cache status
    print("🔧 DEBUG GEMINI: Available tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:50]}...")
    print(f"🔧 DEBUG GEMINI: Dataset path: {dataset_path}")
    print(f"🔧 DEBUG GEMINI: Cache keys: {list(DATAFRAME_CACHE.keys())}")
    
    # Extract filename for prompt
    filename = os.path.basename(dataset_path)
    
    # Format the prompt with the filename
    formatted_prompt = prompt_template.partial(uploaded_filename=filename)
    
    agent = create_openai_tools_agent(llm, tools, formatted_prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
