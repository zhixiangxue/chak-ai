"""
chak.tools.exec - Built-in execution tools

Provides general-purpose execution utilities that can be handed directly
to an LLM agent as callable tools:

  - Bash: execute arbitrary shell commands (cross-platform)
  - Python: execute Python code snippets via the active venv interpreter
"""

from .bash import Bash
from .python import Python

__all__ = ["Bash", "Python"]
