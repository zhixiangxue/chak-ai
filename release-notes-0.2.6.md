# Release Notes - v0.2.6

## 🎯 What's New

### Enhanced Event Stream API

We've improved the event stream experience to give you more control and better type safety when observing tool calls.

**Complete Tool Call Lifecycle**:
```python
from chak.message import MessageChunk, ToolCallStartEvent, ToolCallSuccessEvent, ToolCallErrorEvent

async for event in await conv.asend("Calculate 15 + 27", event=True):
    match event:
        case ToolCallStartEvent(tool_name=name, arguments=args):
            print(f"🔧 Calling: {name} with {args}")
        
        case ToolCallSuccessEvent(tool_name=name, result=res):
            print(f"✅ Result: {name} -> {res}")
        
        case ToolCallErrorEvent(tool_name=name, error=err):
            print(f"❌ Failed: {name} - {err}")
        
        case MessageChunk(content=text, is_final=final, metadata=meta, final_message=msg):
            print(text, end="", flush=True)
            if final and meta:
                print(f"\nTokens used: {meta.get('usage')}")
```

**Key Features**:
- 🎯 **Explicit success/failure events** - `ToolCallSuccessEvent` and `ToolCallErrorEvent` make it clear what happened
- 📊 **Richer content events** - Access `metadata` and `final_message` from `MessageChunk`
- 🔗 **Track tool calls** - Use `call_id` to correlate start/end events
- 🛡️ **Type-safe** - Better match-case pattern matching with distinct event types

## 📚 Learn More

Check out `examples/event_stream_chat_demo.py` for a complete working example.

## ⚠️ Breaking Changes

If you're using the event stream API, you'll need to update your imports:

**Update your imports**:
```python
# Old
from chak.message import ContentEvent, ToolCallEndEvent

# New
from chak.message import MessageChunk, ToolCallSuccessEvent, ToolCallErrorEvent
```

**Update your event handling**:
```python
# Old
case ContentEvent(content=text, is_final=final):
    ...

# New
case MessageChunk(content=text, is_final=final):
    ...  # Plus you can now access metadata and final_message!
```

```python
# Old
case ToolCallEndEvent(success=True, result=res):
    ...
case ToolCallEndEvent(success=False, error=err):
    ...

# New
case ToolCallSuccessEvent(result=res):
    ...
case ToolCallErrorEvent(error=err):
    ...
```
