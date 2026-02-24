# Agents

Working through the examples and concepts from [**Designing Multiagent Systems**](https://github.com/victordibia/designing-multiagent-systems/) by Victor Dibia.

## Contents

- **`agent.py`** - A simple agent using the `picoagents` library with tool calling (weather example)
- **`pseudo-code.py`** - Pseudocode for an agent execution loop illustrating the core run-tool-update cycle
- **`picoframework/`** - A from-scratch mini agent framework exploring the key abstractions:
  - `base_agent.py` - Abstract base class defining the agent interface
  - `agent.py` - Concrete agent implementation with streaming, tool execution, and memory
  - `model_client.py` - LLM provider abstraction with an OpenAI implementation
  - `tools.py` - Tool interface and function-to-tool conversion
  - `context.py` - Conversation context management
  - `memory.py` - Memory abstraction for agent recall

## Attribution

This repository contains my notes and implementations based on Victor Dibia's [Designing Multiagent Systems](https://github.com/victordibia/designing-multiagent-systems/). All credit for the original concepts, architecture patterns, and teaching material goes to the author.
