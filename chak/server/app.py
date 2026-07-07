"""
Chak Server Application

FastAPI-based WebSocket and HTTP gateway for Chak AI SDK.
"""

import json
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .config import ServerConfig
from .websocket import ConversationWebSocketHandler


def create_app(config: ServerConfig) -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        config: Server configuration
        
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Chak AI Gateway",
        description="WebSocket and HTTP gateway for Chak AI SDK",
        version="0.1.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # WebSocket handler
    ws_handler = ConversationWebSocketHandler(config)
    
    @app.websocket("/ws/conversation")
    async def websocket_conversation(websocket: WebSocket):
        """
        WebSocket endpoint for conversation.
        
        Each connection creates a new Conversation object in memory.
        Connection lifecycle = Conversation lifecycle.
        """
        await ws_handler.handle(websocket)
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Chak Server</title>
        </head>
        <body>
            <div>
                🚀 Chak server is running. 
                <a href="/playground">Open Playground</a> | 
                <a href="https://github.com/zhixiangxue/chak-ai" target="_blank">Give Chak a star on GitHub ⭐🤗🥰</a>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    @app.get("/playground")
    async def playground():
        """Interactive playground for testing WebSocket."""
        from pathlib import Path
        from ..utils.models import PROVIDER_MODELS
        
        # Get all provider entries (including custom base_url ones)
        provider_entries = config.get_provider_entries()
        
        # Build provider list and models mapping for frontend
        # For simple providers: just provider name
        # For custom base_url: use full "provider@base_url" as key
        validated_providers = list(provider_entries.keys())
        
        # Build models mapping
        # For "provider@base_url" entries, extract provider name to get models
        provider_models_mapping = {}
        for config_key in validated_providers:
            # Extract base provider name
            if '@' in config_key:
                provider_name = config_key.split('@')[0]
            else:
                provider_name = config_key
            
            # Get models for this provider
            if provider_name in PROVIDER_MODELS:
                provider_models_mapping[config_key] = PROVIDER_MODELS[provider_name]
        
        html_path = Path(__file__).parent / "playground.html"
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Inject providers and models into HTML
        html_content = html_content.replace(
            'let providers = [];',
            f'let providers = {json.dumps(validated_providers)};'
        )
        html_content = html_content.replace(
            'let providerModels = {};',
            f'let providerModels = {json.dumps(provider_models_mapping)};'
        )
        
        return HTMLResponse(content=html_content)
    
    return app


# Configuration file template
CONFIG_TEMPLATE = """\
providers:
  # Simple provider (uses default base_url)
  openai:
    api_key: ${OPENAI_API_KEY}
  
  # Provider with custom base_url
  ollama:
    api_key: "ollama"
    base_url: "http://localhost:11434/v1"
  
  # More providers (uncomment as needed):
  # anthropic:
  #   api_key: ${ANTHROPIC_API_KEY}
  # deepseek:
  #   api_key: ${DEEPSEEK_API_KEY}
  # bailian:
  #   api_key: ${BAILIAN_API_KEY}
  # google:
  #   api_key: ${GOOGLE_API_KEY}
  # zhipu:
  #   api_key: ${ZHIPU_API_KEY}
  # moonshot:
  #   api_key: ${MOONSHOT_API_KEY}
  # minimax:
  #   api_key: ${MINIMAX_API_KEY}
  # mistral:
  #   api_key: ${MISTRAL_API_KEY}
  # xai:
  #   api_key: ${XAI_API_KEY}

server:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
"""


def serve(config_path: Optional[str] = None):
    """
    Start Chak server.
    
    Args:
        config_path: Path to YAML configuration file (required)
        
    Example:
        >>> import chak
        >>> chak.serve("chak-config.yaml")
    """
    # Check if config_path is provided
    if not config_path:
        print("\n❌ Missing configuration file\n")
        print("Usage: chak.serve('path/to/config.yaml')\n")
        print("Configuration file format:\n")
        print("="*60)
        print(CONFIG_TEMPLATE)
        print("="*60)
        return
    
    try:
        # Load configuration
        config = ServerConfig.from_yaml(config_path)
    except FileNotFoundError:
        print(f"\n❌ Configuration file not found: {config_path}\n")
        print("Please create a configuration file with the following format:\n")
        print("="*60)
        print(CONFIG_TEMPLATE)
        print("="*60)
        print("\nThen run: chak.serve('chak-config.yaml')\n")
        return
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}\n")
        print("Expected configuration format:\n")
        print("="*60)
        print(CONFIG_TEMPLATE)
        print("="*60)
        return
    
    # Create app
    app = create_app(config)
    
    # Determine display host (use localhost instead of 0.0.0.0)
    display_host = "localhost" if config.server.host in ["0.0.0.0", "::" ] else config.server.host
    base_url = f"http://{display_host}:{config.server.port}"
    
    # Print beautiful startup banner
    print("\n" + "="*70)
    print("")
    print("  ✨ Chak AI Gateway")
    print("  A simple, yet handy, LLM gateway")
    print("")
    print("="*70)
    print("")
    print(f"  🚀 Server running at:     {base_url}")
    print(f"  🎮 Playground:            {base_url}/playground")
    print(f"  📡 WebSocket endpoint:    ws://{display_host}:{config.server.port}/ws/conversation")
    print("")
    print(f"  ⭐ Star on GitHub:        https://github.com/zhixiangxue/chak-ai")
    print("")
    print("="*70)
    print("")
    
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="warning"  # Reduce log verbosity for cleaner output
    )
