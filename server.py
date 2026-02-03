from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from groq import AsyncGroq


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Groq client
groq_client = AsyncGroq(api_key=os.environ.get('GROQ_API_KEY'))

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]

# System message for the AI market analyst
SYSTEM_MESSAGE = """Act as a local market analyst, small-business mentor, and demand interpreter.

Your task is to explain what business or service demand exists in a specific location using public demand signals, not assumptions.

When analyzing:
1. Analyze the signals and identify real demand patterns
2. Group them into clear business or service categories
3. Rank demand as High / Medium / Low
4. Explain WHY this demand exists in that location (lifestyle, population, work culture, problems)
5. Mention competition level honestly (Low / Medium / High)
6. Suggest:
   • Who this business is suitable for
   • Whether it is online or offline
   • Risk level (Low / Medium / High)

OUTPUT RULES:
- Use very simple language (for beginners)
- Calm, mentor-like tone
- Bullet points
- No hype, no profit promises
- Do not invent fake data or exact numbers
- Use phrases like "based on public search signals" or "visible local activity"

IMPORTANT:
Add a short disclaimer that this reflects demand signals, not guaranteed success.

GOAL:
Reduce confusion and help people make better decisions, not get rich fast."""

@api_router.get("/")
async def root():
    return {"message": "Market Demand Analyzer API - Powered by Groq"}

@api_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Save user message
        user_message = ChatMessage(
            session_id=session_id,
            role="user",
            content=request.message
        )
        
        user_doc = user_message.model_dump()
        user_doc['timestamp'] = user_doc['timestamp'].isoformat()
        await db.chat_messages.insert_one(user_doc)
        
        # Get chat history for context
        history = await db.chat_messages.find(
            {"session_id": session_id},
            {"_id": 0}
        ).sort("timestamp", 1).limit(10).to_list(10)
        
        # Build messages array for Groq
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
        
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Call Groq API
        chat_completion = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=2048
        )
        
        ai_response = chat_completion.choices[0].message.content
        
        # Save bot message
        bot_message = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=ai_response
        )
        
        bot_doc = bot_message.model_dump()
        bot_doc['timestamp'] = bot_doc['timestamp'].isoformat()
        await db.chat_messages.insert_one(bot_doc)
        
        return ChatResponse(
            response=ai_response,
            session_id=session_id,
            timestamp=bot_message.timestamp
        )
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/chat/history/{session_id}", response_model=ConversationHistory)
async def get_chat_history(session_id: str):
    messages = await db.chat_messages.find(
        {"session_id": session_id},
        {"_id": 0}
    ).sort("timestamp", 1).to_list(1000)
    
    for msg in messages:
        if isinstance(msg['timestamp'], str):
            msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
    
    return ConversationHistory(
        session_id=session_id,
        messages=messages
    )

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()