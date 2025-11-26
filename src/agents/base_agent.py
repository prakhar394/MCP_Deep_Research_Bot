from typing import Any, Dict, List, Optional
from ..utils.mcp_schema import MCPMessage, MessageType
from ..utils.logger import get_logger


class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(name)

    async def send_message(
        self,
        message_type: MessageType,
        context_id: str,
        content: Dict[str, Any],
        confidence_score: float = 0.0,
        evidence_uris: Optional[List[str]] = None,
        parent_message_id: Optional[str] = None,
    ) -> MCPMessage:
        msg = MCPMessage(
            message_type=message_type,
            sender_agent=self.name,
            context_id=context_id,
            content=content,
            confidence_score=confidence_score,
            evidence_uris=evidence_uris or [],
            parent_message_id=parent_message_id,
        )
        self.logger.info(f"Sending {message_type} message ({msg.message_id})")
        return msg
