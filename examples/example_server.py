"""
Chak Server Example

This example shows how to start chak as a local gateway service.
Use this when you want to:
  - Keep API keys in a central config file
  - Call LLMs from any programming language via WebSocket
  - Use the Playground UI to test models

Prerequisites:
    1. Create a config file (see server-config.yaml)
    2. Set your API keys in environment variables or config file
    
Usage:
    python examples/example_server.py
"""

import chak

# Start the server with configuration file
# The config file contains all API keys and server settings
chak.serve('examples/server-config.yaml')

# That's it! The server will keep running until you stop it (Ctrl+C)
#
# The Playground UI allows you to:
#   - Select any configured provider
#   - Choose available models
#   - Chat with streaming responses
#   - See conversation history
#
# WebSocket API can be called from any language:
#   - JavaScript/TypeScript
#   - Go
#   - Java
#   - Rust
#   - Python
#   - etc.
