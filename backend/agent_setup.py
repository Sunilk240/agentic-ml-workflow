from langchain_groq import ChatGroq  # Updated import
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
from clustering import perform_clustering, analyze_clusters
from session_management import end_session, show_session_status
from general_assistant import answer_ml_question, explain_pipeline_concept

def create_natural_agent():
    """Creates an agent with natural, high-level tools only."""
    
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
        evaluate_model,
        compare_models,
        suggest_improvements,
        perform_hyperparameter_tuning,
        execute_hyperparameter_tuning,
        auto_tune_hyperparameters,
        perform_clustering,
        analyze_clusters,
        end_session,
        show_session_status,
        answer_ml_question,
        explain_pipeline_concept
    ]

    # Natural conversation prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a friendly, conversational ML assistant that helps users build machine learning pipelines. "
         "You have access to tools for data exploration, cleaning, feature engineering, model training, and evaluation.\n\n"
         "CORE PRINCIPLES:\n"
         "- Be conversational and ask clarifying questions when user input is unclear\n"
         "- Always use the available tools to perform tasks. Never try to execute code directly\n"
         "- When users give incomplete commands, help them find the right command\n"
         "- Provide educational explanations when users ask about ML concepts\n\n"
         "Available capabilities:\n"
         "• Data exploration and analysis\n"
         "• Data cleaning and quality improvement\n"
         "• Feature engineering and transformation\n"
         "• Problem type detection (classification/regression/clustering)\n"
         "• Model training with multiple algorithms\n"
         "• Model evaluation and comparison\n"
         "• Hyperparameter tuning and optimization\n"
         "• Session management and progress tracking\n"
         "• Educational explanations of ML concepts\n\n"
         "When users ask questions, determine if they want to:\n"
         "1. Perform a pipeline operation (use appropriate tool)\n"
         "2. Learn about ML concepts (provide educational explanations)\n"
         "3. Check progress or manage session (use session tools)\n\n"
         "IMPORTANT BEHAVIOR:\n"
         "- If a user input is ambiguous or incomplete (like just 'train', 'clean', 'evaluate'), "
         "ask clarifying questions instead of assuming what they mean\n"
         "- Provide friendly, educational responses instead of technical errors\n"
         "- When unsure about user intent, offer specific options and ask them to choose\n"
         "- Always guide users toward the correct commands\n\n"
         "EXAMPLES of handling ambiguous input:\n"
         "User: 'train' → Response: 'I'd be happy to help with training! Do you want to start the training process? Try: **train model**'\n"
         "User: 'clean' → Response: 'Do you want to analyze and clean your data? Try: **perform cleaning**'\n"
         "User: 'evaluate' → Response: 'Do you want to evaluate trained models? Try: **evaluate model**'\n"
         "User: 'what is X?' → Use educational tools to explain ML concepts\n\n"
         "FALLBACK BEHAVIOR:\n"
         "If no tool matches the user's request, provide a helpful conversational response with:\n"
         "- Acknowledgment of what they said\n"
         "- Clarifying questions or suggestions\n"
         "- List of relevant commands they might want\n"
         "- Offer to explain ML concepts if they seem to be asking educational questions\n\n"
         "Always be helpful, explain your analysis clearly, and guide users through next steps. "
         "Current dataset: {uploaded_filename}"),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    return user_tools, prompt

def initialize_llm():
    """Initialize the LLM with configuration"""
    Config.validate_config()
    
    # Updated to use ChatGroq directly
    return ChatGroq(
        model=Config.MODEL_NAME,
        api_key=Config.GROQ_API_KEY,
        streaming=False,
        temperature=0.1,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

def create_agent_executor(uploaded_filename: str):
    """Create and return the agent executor"""
    llm = initialize_llm()
    tools, prompt_template = create_natural_agent()
    
    # Debug: Print tool names
    print("🔧 DEBUG: Available tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:50]}...")
    print(f"🔧 DEBUG: Dataset filename: {uploaded_filename}")
    
    # Format the prompt with the filename
    formatted_prompt = prompt_template.partial(uploaded_filename=uploaded_filename)
    
    agent = create_openai_tools_agent(llm, tools, formatted_prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
