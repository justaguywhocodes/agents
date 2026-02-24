# tools.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

class BaseTool(ABC):
    """ Abstract base class for all tools. """
    def __init__(self, name:str, description: str):
        self.name = name
        self.description = description 
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """ JSON schema defining expected inputs """
        pass

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """ Execute the tool with given parameters """
        pass

    def to_llm_format(self) -> Dict[str, Any]:
        """ Convert to OpenAI (or any other) function calling format. """
        pass