import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from chak.conversation import Conversation
from chak.tools.manager import ToolCallApproval


app = FastAPI()


def load_openai_api_key() -> str:
    """Load OPENAI_API_KEY from .env using python-dotenv."""
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found after load_dotenv")
    return api_key


def get_current_time() -> str:
    """Get current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


HTML_PAGE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>chak Tool Approval Demo</title>
</head>
<body>
  <h3>chak human-in-the-loop demo</h3>
  <div>
    <input id="user-input" style="width: 400px;"
           value="请用 get_current_time 工具告诉我当前时间" />
    <button id="send-btn">Send</button>
  </div>
  <hr />
  <div id="log" style="white-space: pre-wrap; border: 1px solid #ccc; padding: 8px;
                       width: 600px; height: 300px; overflow-y: auto;"></div>

  <script>
    (function() {
      var log = document.getElementById('log');
      var input = document.getElementById('user-input');
      var sendBtn = document.getElementById('send-btn');

      function append(text) {
        log.textContent += text + '\\n';
        log.scrollTop = log.scrollHeight;
      }

      var ws = new WebSocket('ws://' + window.location.host + '/ws');

      ws.onopen = function() {
        append('[system] WebSocket connected');
      };

      ws.onmessage = function(event) {
        var data = JSON.parse(event.data);

        if (data.type === 'assistant_message') {
          append('Assistant: ' + data.content);
        } else if (data.type === 'tool_approval_required') {
          append('[approval] Model wants to call tool: ' + data.tool_name);
          append('[approval] args: ' + JSON.stringify(data.arguments));

          var allow = window.confirm(
            "Allow tool call '" + data.tool_name + "' ?"
          );
          ws.send(JSON.stringify({
            type: 'tool_approval_decision',
            call_id: data.call_id,
            allow: allow
          }));
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
        append('User: ' + text);
        ws.send(JSON.stringify({
          type: 'user_message',
          content: text
        }));
      };
    })();
  </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    api_key = load_openai_api_key()

    # Per-connection state
    pending_futures: Dict[str, asyncio.Future[bool]] = {}

    async def approval_handler(approval: ToolCallApproval) -> bool:
        """Human-in-the-loop tool approval handler.

        Send approval request to frontend via WebSocket and wait for user decision.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        pending_futures[approval.call_id] = fut

        await websocket.send_json({
            "type": "tool_approval_required",
            "call_id": approval.call_id,
            "tool_name": approval.tool_name,
            "arguments": approval.arguments,
        })

        return await fut

    conv = Conversation(
        model_uri="openai/gpt-4o-mini",
        api_key=api_key,
        tools=[get_current_time],
        tool_approval_handler=approval_handler,
    )

    async def handle_user_message(text: str) -> None:
        try:
            response = await conv.asend(text)
            await websocket.send_json({
                "type": "assistant_message",
                "content": getattr(response, "content", str(response)),
            })
        except Exception as exc:  # pragma: no cover - demo error path
            await websocket.send_json({
                "type": "error",
                "message": f"Conversation error: {exc}",
            })

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            msg_type = data.get("type")

            if msg_type == "user_message":
                content = data.get("content", "")
                asyncio.create_task(handle_user_message(content))

            elif msg_type == "tool_approval_decision":
                call_id = data.get("call_id")
                allow = bool(data.get("allow"))
                fut = pending_futures.pop(call_id, None)
                if fut is not None and not fut.done():
                    fut.set_result(allow)

            else:
                await websocket.send_json({
                    "type": "info",
                    "message": f"Unknown message type: {msg_type}",
                })
    except WebSocketDisconnect:
        conv.close()
    except Exception:
        conv.close()
        raise


if __name__ == "__main__":
    uvicorn.run(
        "examples.tool_approval_web_demo:app",
        host="127.0.0.1",
        port=8888,
        reload=True,
    )
