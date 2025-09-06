import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for ML pipeline"""
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    MODEL_NAME = "llama-3.3-70b-versatile"
    MODEL_PROVIDER = "groq"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    MAX_ITER = 1000
    
    # Thresholds
    OUTLIER_THRESHOLD = 1.5
    HIGH_CARDINALITY_THRESHOLD = 0.5
    CORRELATION_THRESHOLD = 0.8
    SCALING_THRESHOLD = 1000
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        return True
    
    @classmethod
    def validate_gemini_config(cls):
        """Validate that Gemini configuration is present"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        return True
