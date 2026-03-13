from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
from services.analyzer import analyzer
from services.chatbot import bot  

app = FastAPI(title="DermaAI Backend")

# Enable CORS for React Native connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
async def predict_skin_issue(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file.")
    
    try:
        file_bytes = await file.read()
        # Process image using your analyzer service
        analysis_result = await analyzer.process_image(file_bytes)
        return analysis_result
    except Exception as e:
        print(f"Analysis Error: {e}")
        raise HTTPException(status_code=500, detail="Image analysis failed.")