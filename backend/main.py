from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
from services.analyzer import analyzer
from services.chatbot import bot  
from sqlalchemy.orm import Session
from database import SessionLocal, ScanResult, init_db, get_db

app = FastAPI(title="DermaAI Backend")

# Enable CORS for React Native connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "online"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Pass the question to your bot service
        response = bot.ask(request.prompt)
        return {"reply": response}
    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail="Chatbot error.")
    
@app.post("/predict")
async def predict_skin_issue(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db) 
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file.")
    
    try:
        # This line creates the 'file_bytes' variable that was missing!
        file_bytes = await file.read() 
        
        # Process image
        analysis_result = await analyzer.process_image(file_bytes)
        
        # Save to Database
        db_record = ScanResult(
            condition=analysis_result["condition"],
            confidence=analysis_result["confidence"],
            description=analysis_result["description"]
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        return analysis_result
        
    except Exception as e:
        print(f"Analysis Error: {e}")
        raise HTTPException(status_code=500, detail="Image analysis failed.")