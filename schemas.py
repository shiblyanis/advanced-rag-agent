from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    user_id: str = "demo-user"
    session_id: str = "default"

class Source(BaseModel):
    filename: str
    page: Optional[int] = None
    score: Optional[float] = None
    text: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    latency_ms: float
    retrieved: int = 0
    reranked: int = 0
    memory_used: bool = False

class MemoryCreate(BaseModel):
    user_id: str = "demo-user"
    text: str = Field(min_length=1, max_length=4000)
