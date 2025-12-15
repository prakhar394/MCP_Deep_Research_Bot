from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime


class MessageType(str, Enum):
    QUERY = "query"
    RETRIEVAL = "retrieval"
    SUMMARY = "summary"
    VERIFICATION = "verification"
    ERROR = "error"


class MCPMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType
    sender_agent: str
    context_id: str
    content: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    confidence_score: float = 0.0
    parent_message_id: Optional[str] = None
    evidence_uris: Optional[List[str]] = None
