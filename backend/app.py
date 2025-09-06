from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from typing import Optional
import uuid

# Import your existing agent
from agent_setup_gemini import create_agent_executor_gemini

app = FastAPI(title="ML Agent API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    detailed_data: Optional[dict] = None

# Global variables for session management
sessions = {}
UPLOAD_DIR = "uploads"
last_tool_detailed_data = {}

# Create upload directory
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "ML Agent API is running"}

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload dataset file"""
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize session
        sessions[session_id] = {
            "dataset_path": file_path,
            "filename": file.filename,
            "agent": None
        }
        
        return {
            "message": f"Dataset {file.filename} uploaded successfully",
            "session_id": session_id,
            "filename": file.filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(chat_data: ChatMessage):
    """Chat with ML agent"""
    try:
        session_id = chat_data.session_id
        
        if not session_id or session_id not in sessions:
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        # Initialize agent if not exists
        if sessions[session_id]["agent"] is None:
            # Pass the actual uploaded file path to the agent
            dataset_path = sessions[session_id]["dataset_path"]
            sessions[session_id]["agent"] = create_agent_executor_gemini(dataset_path)
        
        agent = sessions[session_id]["agent"]
        
        # Get response from agent
        response = agent.invoke({"input": chat_data.message})
        
        # Check if there's detailed data from the last tool execution
        from helpers import PIPELINE_STATE
        detailed_data = PIPELINE_STATE.get("last_detailed_data", None)
        
        # Clear the detailed data after using it
        if "last_detailed_data" in PIPELINE_STATE:
            del PIPELINE_STATE["last_detailed_data"]
        
        return ChatResponse(
            response=response["output"],
            session_id=session_id,
            detailed_data=detailed_data
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/session/{session_id}/info")
async def get_session_info(session_id: str):
    """Get session information"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "filename": session.get("filename"),
        "dataset_uploaded": session.get("dataset_path") is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)