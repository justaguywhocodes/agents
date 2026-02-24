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
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None       
    ) -> None:
        """ Add new memory, removing oldest if at capacity"""
        memory_item = MemoryItem(
            content=content, 
            metadata=metadata or {}
        )
        self.memories.append(memory_item)

        # Remove oldest memories if over capacity
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.memories:]
    
    async def query(
            self,
            query: str,
            max_results: int = 10
    ) -> List[str]:
        """ Simple text-based search in memory contents."""
        query_lower = query.lower()
        matching_memories = []

        for memory in reversed(self.memories): # Search newest first
            if query_lower in memory.content.lower():
                matching_memories.append(memory.content)
                if len(matching_memories) >= max_results:
                    break
        return matching_memories
    
    async def get_context(self, max_items: int = 10) -> List[str]:
        """ Get most recent memories for context"""
        recent_memories = (
            self.memories[-max_items:]  if self.memories else []
        )
        return [memory.content for memory in recent_memories]
    
class MemoryBackend:
    def _validate_path(self, path: str) -> Path:
        """ Prepare directory traversal attacks."""
        # Resolve to absolute path
        full_path = (self.base_path / path).resolve()

        # Ensure path stays within base directory
        try:
            full_path.relative_to(self.base_path)
        except ValueError:
            raise ValueError(
                "Access denied: path outside memory"
            )
        return full_path
