"""
Custom payload + LLM + tool calling demo.

Run:
    python examples/custom_payload_demo.py

This example shows the full workflow:
1. User sends a message to LLM
2. LLM decides to call a tool
3. Tool returns structured data describing a form
4. Server wraps the form data into custom
5. Frontend receives the message and dynamically renders the form

Prerequisites:
    export OPENAI_API_KEY=sk-your-key-here
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from chak.conversation import Conversation
from chak.message import AIMessage


app = FastAPI()


def load_openai_api_key() -> str:
    """Load OPENAI_API_KEY from .env using python-dotenv."""
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found after load_dotenv")
    return api_key


class FormField(BaseModel):
    """Single form field definition."""
    name: str
    label: str
    type: str


class FormSchema(BaseModel):
    """Structured form schema returned by tool."""
    type: str
    title: str
    fields: list[FormField]


def get_user_info_form() -> dict:
    """Return a form schema dict for collecting user info.
    
    This tool simulates a backend service that generates form definitions.
    The LLM can call this tool when it needs to collect user information.
    
    Returns a dict that will be used as message.custom.
    """
    return {
        "type": "form_v1",
        "title": "用户信息",
        "fields": [
            {"name": "name", "label": "姓名", "type": "text"},
            {"name": "age", "label": "年龄", "type": "number"},
            {"name": "email", "label": "邮箱", "type": "email"},
        ]
    }


HTML_PAGE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>chak custom + LLM demo</title>
</head>
<body>
  <h3>chak custom + LLM + tool calling demo</h3>
  <div style="margin-bottom: 10px;">
    <label><input type="radio" name="mode" value="stream" checked> Streaming</label>
    <label style="margin-left: 15px;"><input type="radio" name="mode" value="nonstream"> Non-Streaming</label>
  </div>
  <div>
    <input id="user-input" style="width: 400px;" value="请给我一个表单收集用户信息" />
    <button id="send-btn">Send</button>
  </div>
  <hr />
  <div id="log" style="white-space: pre-wrap; border: 1px solid #ccc; padding: 8px; width: 600px; height: 200px; overflow-y: auto;"></div>
  <div id="form-container"></div>

  <script>
    (function() {
      var log = document.getElementById('log');
      var input = document.getElementById('user-input');
      var sendBtn = document.getElementById('send-btn');
      var formContainer = document.getElementById('form-container');

      function append(text) {
        log.textContent += text + '\\n';
        log.scrollTop = log.scrollHeight;
      }

      function renderForm(payload) {
        formContainer.innerHTML = '';
        if (!payload || payload.type !== 'form_v1') {
          return;
        }

        var form = document.createElement('form');

        if (payload.title) {
          var title = document.createElement('h4');
          title.textContent = payload.title;
          form.appendChild(title);
        }

        if (Array.isArray(payload.fields)) {
          payload.fields.forEach(function(field) {
            var label = document.createElement('label');
            label.textContent = field.label || field.name || '';
            var input = document.createElement('input');
            input.name = field.name || '';
            input.type = field.type || 'text';

            form.appendChild(label);
            form.appendChild(document.createElement('br'));
            form.appendChild(input);
            form.appendChild(document.createElement('br'));
          });
        }

        var submitBtn = document.createElement('button');
        submitBtn.type = 'button';
        submitBtn.textContent = 'Submit';
        submitBtn.onclick = function() {
          var result = {};
          var inputs = form.querySelectorAll('input');
          for (var i = 0; i < inputs.length; i++) {
            result[inputs[i].name] = inputs[i].value;
          }
          append('[client] form values: ' + JSON.stringify(result));
          alert('Form values: ' + JSON.stringify(result));
        };
        form.appendChild(submitBtn);

        formContainer.appendChild(form);
      }

      var ws = new WebSocket('ws://' + window.location.host + '/ws');

      ws.onopen = function() {
        append('[system] WebSocket connected');
      };

      ws.onmessage = function(event) {
        var data = JSON.parse(event.data);
        if (data.type === 'chunk') {
          // Streaming content chunk - append without newline
          log.textContent += data.content;
          log.scrollTop = log.scrollHeight;
        } else if (data.type === 'stream_end') {
          // Stream finished, add newline and check for custom
          log.textContent += '\\n';
          log.scrollTop = log.scrollHeight;
          sendBtn.disabled = false;
          sendBtn.textContent = 'Send';
          if (data.custom && Object.keys(data.custom).length > 0) {
            append('[system] Got custom, rendering form...');
            renderForm(data.custom);
          }
        } else if (data.type === 'assistant_message') {
          append('Assistant: ' + data.content);
          sendBtn.disabled = false;
          sendBtn.textContent = 'Send';
          if (data.custom && Object.keys(data.custom).length > 0) {
            append('[system] Got custom, rendering form...');
            renderForm(data.custom);
          }
        } else if (data.type === 'info') {
          append('[info] ' + data.message);
        } else if (data.type === 'error') {
          append('[error] ' + data.message);
        }
      };

      ws.onclose = function() {
        append('[system] WebSocket disconnected');
      };

      sendBtn.onclick = function() {
        var text = input.value.trim();
        if (!text) return;
        
        // Get selected mode
        var modeRadios = document.getElementsByName('mode');
        var streamMode = true;
        for (var i = 0; i < modeRadios.length; i++) {
          if (modeRadios[i].checked) {
            streamMode = modeRadios[i].value === 'stream';
            break;
          }
        }
        
        append('User: ' + text);
        if (streamMode) {
          append('Assistant: ');  // Prepare for streaming
        }
        sendBtn.disabled = true;
        sendBtn.textContent = 'Sending...';
        ws.send(JSON.stringify({
          type: 'user_message',
          content: text,
          stream: streamMode
        }));
      };
    })();
  </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
  """Serve minimal HTML page."""
  return HTMLResponse(HTML_PAGE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
  """Handle WebSocket connection with LLM + tool calling."""
  await websocket.accept()
  
  api_key = load_openai_api_key()
  
  # Create conversation with the form-generation tool
  conv = Conversation(
    model_uri="openai/gpt-4o-mini",
    api_key=api_key,
    system_message="你是一个助手。当用户需要表单时，使用 get_user_info_form 工具获取表单定义。",
    tools=[get_user_info_form],
  )
  
  try:
    while True:
      raw = await websocket.receive_text()
      data = json.loads(raw)
      msg_type = data.get("type")

      if msg_type == "user_message":
        content = data.get("content", "")
        stream_mode = data.get("stream", True)  # Default to streaming
        
        if stream_mode:
          # Streaming mode: real-time chunks + custom at end
          stream = await conv.asend(content, stream=True)
          
          async for chunk in stream:
            if chunk.content:
              await websocket.send_json({
                "type": "chunk",
                "content": chunk.content,
              })
          
          # Extract custom from ToolMessage in conversation history
          custom_data = None
          for msg in reversed(conv.messages):
            if hasattr(msg, 'role') and msg.role == 'tool':
              try:
                tool_result = json.loads(msg.content)
                if isinstance(tool_result, dict) and tool_result.get("type") == "form_v1":
                  custom_data = tool_result
                  break
              except (json.JSONDecodeError, AttributeError):
                pass
          
          await websocket.send_json({
            "type": "stream_end",
            "custom": custom_data or {},
          })
        else:
          # Non-streaming mode: complete message at once
          response = await conv.asend(content, stream=False)
          
          # Extract custom from ToolMessage
          custom_data = None
          for msg in reversed(conv.messages):
            if hasattr(msg, 'role') and msg.role == 'tool':
              try:
                tool_result = json.loads(msg.content)
                if isinstance(tool_result, dict) and tool_result.get("type") == "form_v1":
                  custom_data = tool_result
                  break
              except (json.JSONDecodeError, AttributeError):
                pass
          
          await websocket.send_json({
            "type": "assistant_message",
            "content": response.content if hasattr(response, 'content') else str(response),
            "custom": custom_data or {},
          })
      else:
        await websocket.send_json({
          "type": "info",
          "message": f"Unknown message type: {msg_type}",
        })
  except WebSocketDisconnect:
    conv.close()
  except Exception as e:
    conv.close()
    await websocket.send_json({
      "type": "error",
      "message": f"Error: {str(e)}",
    })
    raise


if __name__ == "__main__":
  uvicorn.run(
    "examples.custom_payload_demo:app",
    host="127.0.0.1",
    port=8889,
    reload=True,
  )
