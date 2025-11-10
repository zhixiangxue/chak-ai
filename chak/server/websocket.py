"""
WebSocket handler for conversation endpoint.
"""

import json
import traceback
from typing import Dict, Optional, Any

from fastapi import WebSocket, WebSocketDisconnect

from .config import ServerConfig
from .schemas import (
    InitMessage, SendMessage, AddMessagesMessage,
    StatsResponse, ErrorResponse
)
from ..context.strategies import NoopStrategy, FIFOStrategy
from ..conversation import Conversation


class ConversationWebSocketHandler:
    """WebSocket handler for /ws/conversation endpoint."""
    
    def __init__(self, config: ServerConfig):
        """
        Initialize handler.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.conversations: Dict[int, Conversation] = {}
    
    async def handle(self, websocket: WebSocket):
        """
        Handle WebSocket connection.
        
        Args:
            websocket: WebSocket connection
        """
        await websocket.accept()
        conn_id = id(websocket)
        conversation: Optional[Conversation] = None
        
        try:
            while True:
                try:
                    # Receive message as text
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    if msg_type == "init":
                        conversation = await self._handle_init(websocket, data, conn_id)
                    
                    elif msg_type == "send":
                        if not conversation:
                            await self._send_error(websocket, "Conversation not initialized")
                            continue
                        await self._handle_send(websocket, conversation, data)
                    
                    elif msg_type == "add_messages":
                        if not conversation:
                            await self._send_error(websocket, "Conversation not initialized")
                            continue
                        await self._handle_add_messages(websocket, conversation, data)
                    
                    elif msg_type == "reset":
                        if not conversation:
                            await self._send_error(websocket, "Conversation not initialized")
                            continue
                        conversation.reset()
                        await websocket.send_text(json.dumps({"type": "ok", "action": "reset"}))
                    
                    elif msg_type == "clear":
                        if not conversation:
                            await self._send_error(websocket, "Conversation not initialized")
                            continue
                        conversation.clear()
                        await websocket.send_text(json.dumps({"type": "ok", "action": "clear"}))
                    
                    elif msg_type == "stats":
                        if not conversation:
                            await self._send_error(websocket, "Conversation not initialized")
                            continue
                        stats = conversation.stats()
                        response = StatsResponse(data=stats)
                        await websocket.send_text(response.model_dump_json())
                    
                    else:
                        await self._send_error(websocket, f"Unknown message type: {msg_type}")
                
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON format")
                except Exception as e:
                    await self._send_error(websocket, str(e), traceback.format_exc())
        
        except WebSocketDisconnect:
            pass
        except Exception as e:
            # Handle any other errors (e.g., client disconnect)
            pass
        
        finally:
            # Cleanup: destroy conversation object when connection closes
            if conn_id in self.conversations:
                conv = self.conversations[conn_id]
                conv.close()
                del self.conversations[conn_id]
    
    async def _handle_init(
        self,
        websocket: WebSocket,
        data: dict,
        conn_id: int
    ) -> Conversation:
        """Handle init message."""
        msg = InitMessage(**data)
        
        # Extract provider from model_uri
        provider = self._extract_provider(msg.model_uri)
        
        # Get provider config (API key + optional base_url)
        provider_config = self.config.get_provider_config(provider)
        if not provider_config:
            await self._send_error(
                websocket,
                f"API key not configured for provider: {provider}"
            )
            raise ValueError(f"API key not found for provider: {provider}")
        
        # Use model_uri as-is from frontend
        # Frontend already constructs the correct format:
        # - Simple: provider/model or provider:model
        # - Custom base_url: provider@base_url:model
        final_model_uri = msg.model_uri
        
        # Create context strategy
        context_strategy = self._create_strategy(msg.context_strategy)
        
        # Create Conversation (same as SDK!)
        conversation = Conversation(
            model_uri=final_model_uri,
            api_key=provider_config['api_key'],
            system_message=msg.system_message,
            context_strategy=context_strategy
        )
        
        self.conversations[conn_id] = conversation
        
        await websocket.send_text(json.dumps({
            "type": "ok",
            "action": "init",
            "model_uri": final_model_uri
        }))
        
        return conversation
    
    async def _handle_send(
        self,
        websocket: WebSocket,
        conversation: Conversation,
        data: dict
    ):
        """Handle send message."""
        msg = SendMessage(**data)
        
        # Validate role
        role = msg.role  # type: ignore
        
        if msg.stream:
            # Streaming mode - just forward all chunks as-is
            chunks = conversation.send(
                message=msg.message,
                role=role,
                stream=True
            )
            
            for chunk in chunks:  # type: ignore
                # Build chunk data
                chunk_data = {
                    "type": "chunk",
                    "content": chunk.content,
                    "is_final": chunk.is_final
                }
                
                # Add final_message if present
                if chunk.is_final and chunk.final_message:
                    chunk_data["final_message"] = {
                        "role": chunk.final_message.role,
                        "content": chunk.final_message.content,
                        "metadata": self._serialize_metadata(chunk.final_message.metadata)
                    }
                
                # Send it immediately
                await websocket.send_text(json.dumps(chunk_data))
        else:
            # Non-streaming mode
            response_msg = conversation.send(
                message=msg.message,
                role=role,
                stream=False
            )
            
            # Build response manually
            response = {
                "type": "message",
                "message": {
                    "role": response_msg.role,  # type: ignore
                    "content": response_msg.content,  # type: ignore
                    "metadata": self._serialize_metadata(response_msg.metadata)  # type: ignore
                }
            }
            await websocket.send_text(json.dumps(response))
    
    async def _handle_add_messages(
        self,
        websocket: WebSocket,
        conversation: Conversation,
        data: dict
    ):
        """Handle add_messages."""
        msg = AddMessagesMessage(**data)
        conversation.add_messages(msg.messages)  # type: ignore
        await websocket.send_text(json.dumps({
            "type": "ok",
            "action": "add_messages",
            "count": len(msg.messages)
        }))
    
    async def _send_error(
        self,
        websocket: WebSocket,
        error: str,
        detail: Optional[str] = None
    ):
        """Send error response."""
        response = ErrorResponse(error=error, detail=detail)
        await websocket.send_text(response.model_dump_json())
    
    def _extract_provider(self, model_uri: str) -> str:
        """
        Extract provider name from model_uri.
        
        Args:
            model_uri: Model URI (e.g., 'openai:gpt-4', 'bailian@https://...:qwen-plus')
            
        Returns:
            Provider name
        """
        # Simple extraction: get part before ':' or '@'
        if '@' in model_uri:
            return model_uri.split('@')[0]
        elif ':' in model_uri:
            return model_uri.split(':')[0]
        elif '/' in model_uri:
            return model_uri.split('/')[0]
        else:
            return model_uri
    
    def _create_strategy(self, strategy_name: Optional[str]):
        """Create context strategy instance."""
        if not strategy_name or strategy_name == "noop":
            return NoopStrategy()
        elif strategy_name == "fifo":
            return FIFOStrategy()
        # Add more strategies as needed
        else:
            return NoopStrategy()
    
    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize metadata to JSON-compatible format.
        
        Handles special objects like CompletionUsage that can't be directly serialized.
        """
        if not metadata:
            return {}
        
        result = {}
        for key, value in metadata.items():
            if hasattr(value, 'model_dump'):
                # Pydantic model
                result[key] = value.model_dump()
            elif hasattr(value, '__dict__'):
                # Object with __dict__
                result[key] = vars(value)
            else:
                result[key] = value
        
        return result
