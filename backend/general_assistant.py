from typing import Dict, Any
from langchain_core.tools import tool

@tool
def answer_ml_question(question: str) -> Dict[str, Any]:
    """
    Answers general machine learning, data science, and AI questions.
    User asks educational questions like 'what is hyperparameter tuning' or 'explain random forest'.
    """
    
    # Common ML topics and explanations
    ml_explanations = {
        "hyperparameter tuning": {
            "definition": "The process of optimizing the configuration settings of machine learning algorithms to improve model performance.",
            "explanation": "Hyperparameters are settings that control how an algorithm learns (like learning rate, number of trees, etc.). Unlike model parameters that are learned from data, hyperparameters are set before training. Tuning involves testing different combinations to find the best performing configuration.",
            "examples": [
                "Random Forest: n_estimators (number of trees), max_depth (tree depth)",
                "SVM: C (regularization), kernel (linear, rbf, poly), gamma (kernel coefficient)",
                "Neural Networks: learning_rate, batch_size, number_of_layers"
            ],
            "methods": [
                "Grid Search: Test all combinations of predefined values",
                "Random Search: Randomly sample parameter combinations", 
                "Bayesian Optimization: Use probability to guide search",
                "Manual Tuning: Expert knowledge-based adjustment"
            ]
        },
        
        "random forest": {
            "definition": "An ensemble learning method that combines multiple decision trees to make predictions.",
            "explanation": "Random Forest builds many decision trees using random subsets of data and features, then averages their predictions. This reduces overfitting and improves accuracy compared to single decision trees.",
            "advantages": [
                "Handles overfitting well due to averaging",
                "Works with both classification and regression",
                "Provides feature importance rankings",
                "Robust to outliers and missing values"
            ],
            "disadvantages": [
                "Can be slow on large datasets",
                "Less interpretable than single decision trees",
                "May overfit with very noisy data"
            ]
        },
        
        "classification vs regression": {
            "definition": "Two main types of supervised machine learning problems.",
            "classification": {
                "purpose": "Predicts categories or classes",
                "output": "Discrete labels (e.g., spam/not spam, cat/dog/bird)",
                "examples": ["Email spam detection", "Image recognition", "Medical diagnosis"],
                "metrics": ["Accuracy", "Precision", "Recall", "F1-score"]
            },
            "regression": {
                "purpose": "Predicts continuous numerical values",
                "output": "Real numbers (e.g., price, temperature, age)",
                "examples": ["House price prediction", "Stock price forecasting", "Sales prediction"],
                "metrics": ["R² score", "Mean Squared Error", "Mean Absolute Error"]
            }
        },
        
        "overfitting": {
            "definition": "When a model learns the training data too well, including noise and irrelevant patterns.",
            "explanation": "An overfitted model performs excellently on training data but poorly on new, unseen data. It memorizes rather than generalizes.",
            "signs": [
                "High training accuracy, low validation accuracy",
                "Large gap between training and test performance",
                "Model performs poorly on new data"
            ],
            "solutions": [
                "Use more training data",
                "Reduce model complexity",
                "Apply regularization techniques",
                "Use cross-validation",
                "Early stopping in neural networks"
            ]
        },
        
        "cross validation": {
            "definition": "A technique to assess how well a model generalizes to unseen data.",
            "explanation": "Cross-validation splits data into multiple folds, trains on some folds and tests on others, then averages the results. This provides a more robust estimate of model performance.",
            "types": [
                "K-Fold: Split data into k equal parts",
                "Stratified K-Fold: Maintains class distribution in each fold",
                "Leave-One-Out: Use each sample as test set once",
                "Time Series Split: Respects temporal order"
            ]
        },
        
        "feature engineering": {
            "definition": "The process of creating, transforming, and selecting features to improve model performance.",
            "explanation": "Raw data often needs preprocessing before machine learning. Feature engineering involves creating new variables, transforming existing ones, and selecting the most relevant features.",
            "techniques": [
                "Scaling: Normalize feature ranges (StandardScaler, MinMaxScaler)",
                "Encoding: Convert categorical to numerical (One-hot, Label encoding)",
                "Creation: Polynomial features, interaction terms, domain-specific features",
                "Selection: Remove irrelevant or redundant features"
            ]
        },
        
        "ensemble methods": {
            "definition": "Techniques that combine multiple models to create a stronger predictor.",
            "explanation": "Instead of relying on a single model, ensemble methods combine predictions from multiple models to improve accuracy and robustness.",
            "types": [
                "Bagging: Train models on different data subsets (Random Forest)",
                "Boosting: Sequential models that learn from previous mistakes (XGBoost)",
                "Voting: Combine predictions by majority vote or averaging",
                "Stacking: Use a meta-model to combine base model predictions"
            ]
        }
    }
    
    # Normalize question for matching
    question_lower = question.lower()
    
    # Find matching topic
    matched_topic = None
    for topic, content in ml_explanations.items():
        if topic in question_lower or any(word in question_lower for word in topic.split()):
            matched_topic = topic
            break
    
    if matched_topic:
        content = ml_explanations[matched_topic]
        
        response = f"📚 **{matched_topic.title()}**\n\n"
        response += f"**Definition:** {content['definition']}\n\n"
        
        if 'explanation' in content:
            response += f"**Explanation:** {content['explanation']}\n\n"
        
        # Add specific content based on topic
        if 'examples' in content:
            response += "**Examples:**\n"
            for example in content['examples']:
                response += f"• {example}\n"
            response += "\n"
        
        if 'methods' in content:
            response += "**Methods:**\n"
            for method in content['methods']:
                response += f"• {method}\n"
            response += "\n"
        
        if 'advantages' in content:
            response += "**Advantages:**\n"
            for advantage in content['advantages']:
                response += f"• {advantage}\n"
            response += "\n"
        
        if 'disadvantages' in content:
            response += "**Disadvantages:**\n"
            for disadvantage in content['disadvantages']:
                response += f"• {disadvantage}\n"
            response += "\n"
        
        if 'signs' in content:
            response += "**Signs:**\n"
            for sign in content['signs']:
                response += f"• {sign}\n"
            response += "\n"
        
        if 'solutions' in content:
            response += "**Solutions:**\n"
            for solution in content['solutions']:
                response += f"• {solution}\n"
            response += "\n"
        
        if 'types' in content:
            response += "**Types:**\n"
            for type_info in content['types']:
                response += f"• {type_info}\n"
            response += "\n"
        
        if 'techniques' in content:
            response += "**Techniques:**\n"
            for technique in content['techniques']:
                response += f"• {technique}\n"
            response += "\n"
        
        # Handle special cases
        if matched_topic == "classification vs regression":
            response += "**Classification:**\n"
            for key, value in content['classification'].items():
                if isinstance(value, list):
                    response += f"• **{key.title()}:** {', '.join(value)}\n"
                else:
                    response += f"• **{key.title()}:** {value}\n"
            response += "\n**Regression:**\n"
            for key, value in content['regression'].items():
                if isinstance(value, list):
                    response += f"• **{key.title()}:** {', '.join(value)}\n"
                else:
                    response += f"• **{key.title()}:** {value}\n"
            response += "\n"
        
        # Add pipeline connection if relevant
        pipeline_connections = {
            "hyperparameter tuning": "💡 **Try it:** Use 'perform hyperparameter tuning' in our pipeline!",
            "random forest": "💡 **Try it:** Use 'train model' → 'proceed with training random_forest'",
            "classification vs regression": "💡 **Try it:** Use 'train model' to auto-detect your problem type!",
            "feature engineering": "💡 **Try it:** Use 'perform feature engineering' in our pipeline!",
            "clustering": "💡 **Try it:** Use 'perform clustering' to find data patterns!"
        }

        
        if matched_topic in pipeline_connections:
            response += f"{pipeline_connections[matched_topic]}\n\n"
        
        response += "**❓ Have more questions?** Ask about any ML concept!"
        
        return {
            "question_answered": True,
            "topic": matched_topic,
            "formatted_response": response,
            "educational_content": True
        }
    
    else:
        # General ML assistant response for unmatched questions
        response = "🤖 **General ML Assistant**\n\n"
        response += f"**Your Question:** {question}\n\n"
        response += "I can help explain these ML concepts:\n"
        response += "• **Algorithms:** Random Forest, SVM, Neural Networks, etc.\n"
        response += "• **Techniques:** Hyperparameter tuning, Cross-validation, Feature engineering\n"
        response += "• **Concepts:** Overfitting, Bias-variance tradeoff, Ensemble methods\n"
        response += "• **Problem Types:** Classification vs Regression, Supervised vs Unsupervised\n"
        response += "• **Evaluation:** Metrics, Validation techniques, Model selection\n\n"
        
        response += "**💡 EXAMPLES:**\n"
        response += "• 'What is hyperparameter tuning?'\n"
        response += "• 'Explain random forest algorithm'\n"
        response += "• 'Difference between classification and regression'\n"
        response += "• 'How to prevent overfitting?'\n"
        response += "• 'What is cross validation?'\n\n"
        
        response += "**🔧 PIPELINE HELP:**\n"
        response += "• Type 'help' for available pipeline commands\n"
        response += "• Type 'session status' to see current progress\n"
        response += "• Ask specific questions about any ML concept!\n"
        
        return {
            "general_response": True,
            "question": question,
            "formatted_response": response,
            "educational_content": True,
            "suggestions_provided": True
        }

@tool
def explain_pipeline_concept(concept: str) -> Dict[str, Any]:
    """
    Explains concepts specific to our ML pipeline workflow.
    User asks about pipeline stages, tools, or workflow questions.
    """
    
    pipeline_concepts = {
        "ml pipeline": {
            "definition": "A sequence of data processing and machine learning steps that automate the ML workflow.",
            "our_pipeline": [
                "1. Data Exploration - Understand your dataset",
                "2. Data Cleaning - Handle missing values, outliers, duplicates", 
                "3. Feature Engineering - Transform and prepare features",
                "4. Problem Detection - Identify classification/regression/clustering",
                "5. Model Training - Train and compare algorithms",
                "6. Model Evaluation - Assess performance and get insights",
                "7. Model Improvement - Hyperparameter tuning and optimization"
            ],
            "benefits": [
                "Consistent and reproducible results",
                "Automated workflow reduces errors",
                "Easy to experiment with different approaches",
                "Guided process for beginners"
            ]
        },
        
        "data cleaning": {
            "purpose": "Prepare raw data for machine learning by fixing quality issues.",
            "our_approach": [
                "Automatic detection of missing values, duplicates, and outliers",
                "Smart recommendations based on data characteristics",
                "Multiple cleaning strategies with explanations",
                "User choice between recommended and alternative methods"
            ],
            "techniques": [
                "Missing values: Fill with mean/median/mode or remove rows",
                "Duplicates: Keep first/last occurrence or remove all",
                "Outliers: Cap to bounds, remove, or replace with median"
            ]
        },
        
        "problem detection": {
            "purpose": "Automatically identify whether your dataset is suited for classification, regression, or clustering.",
            "how_it_works": [
                "Analyzes target column characteristics (data type, unique values)",
                "Calculates uniqueness ratio and examines sample values",
                "Provides confidence score for the detected problem type",
                "Suggests appropriate algorithms for the identified problem"
            ],
            "problem_types": [
                "Classification: Predicting categories (spam/not spam, disease/healthy)",
                "Regression: Predicting numbers (price, temperature, age)",
                "Clustering: Finding groups in data without labels"
            ]
        }
    }
    
    concept_lower = concept.lower()
    matched_concept = None
    
    for topic, content in pipeline_concepts.items():
        if topic in concept_lower or any(word in concept_lower for word in topic.split()):
            matched_concept = topic
            break
    
    if matched_concept:
        content = pipeline_concepts[matched_concept]
        
        response = f"🔧 **{matched_concept.title()} in Our Pipeline**\n\n"
        response += f"**Purpose:** {content.get('definition', content.get('purpose', ''))}\n\n"
        
        if 'our_pipeline' in content:
            response += "**Our ML Pipeline Steps:**\n"
            for step in content['our_pipeline']:
                response += f"{step}\n"
            response += "\n"
        
        if 'our_approach' in content:
            response += "**Our Approach:**\n"
            for approach in content['our_approach']:
                response += f"• {approach}\n"
            response += "\n"
        
        if 'how_it_works' in content:
            response += "**How It Works:**\n"
            for step in content['how_it_works']:
                response += f"• {step}\n"
            response += "\n"
        
        if 'techniques' in content:
            response += "**Techniques:**\n"
            for technique in content['techniques']:
                response += f"• {technique}\n"
            response += "\n"
        
        if 'benefits' in content:
            response += "**Benefits:**\n"
            for benefit in content['benefits']:
                response += f"• {benefit}\n"
            response += "\n"
        
        if 'problem_types' in content:
            response += "**Problem Types:**\n"
            for ptype in content['problem_types']:
                response += f"• {ptype}\n"
            response += "\n"
        
        response += "**💡 Try it in the pipeline!** Load a dataset and follow the guided workflow."
        
        return {
            "concept_explained": True,
            "concept": matched_concept,
            "formatted_response": response,
            "pipeline_specific": True
        }
    
    else:
        response = f"🔧 **Pipeline Concept Help**\n\n"
        response += f"**Your Question:** {concept}\n\n"
        response += "I can explain these pipeline concepts:\n"
        response += "• **ML Pipeline** - Our complete workflow\n"
        response += "• **Data Cleaning** - How we handle data quality\n"
        response += "• **Problem Detection** - How we identify problem types\n"
        response += "• **Feature Engineering** - How we prepare features\n"
        response += "• **Model Training** - Our two-phase training approach\n\n"
        
        response += "**💡 EXAMPLES:**\n"
        response += "• 'Explain ML pipeline'\n"
        response += "• 'How does data cleaning work?'\n"
        response += "• 'What is problem detection?'\n\n"
        
        response += "**🔧 PIPELINE COMMANDS:**\n"
        response += "• Type 'help' for all available commands\n"
        response += "• Type 'session status' to see current progress\n"
        
        return {
            "general_pipeline_help": True,
            "concept": concept,
            "formatted_response": response,
            "pipeline_specific": True
        }