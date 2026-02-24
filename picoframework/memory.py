# memory.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseMemory(ABC):
    """Abstract interface for agent memory systems."""

    @abstractmethod
    async def add(
        self,
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None: 
        """ Store new content in memory """
        pass

    @abstractmethod
    async def query(self, query: str, limit: int = 10) -> List[str]:
        """Retrieve relevant memories based on query."""
        pass  

    @abstractmethod
    async def get_context(self, max_items: int = 10) -> List[str]:
        """Get recent/relevant context for LLM prompt.""" 
        pass  

class ListMemory(Memory):
    """Simple in-memory list-based memory storage."""

    def __init__(self, max_memories: int = 1000):
        self.memories: List[MemoryItem] = []
        self.max_memories = max_memories
    
    async def add(
            
    )