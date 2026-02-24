from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Exception
import re

@dataclass
class MiddlewareContext:
    operation: str # "model_call" or "tool_call"
    agent_name: str
    agent_context: Any # AgentContext
    data: Any # message or tool call
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseMiddleware(ABC):
    """ Abstract base class for middlware. """

    @abstractmethod
    async def process_request(
        self,
        context: MiddlewareContext
    ) -> MiddlewareContext:
        """ Process the operation before execution. """
        pass

    @abstractmethod
    async def process_response(
        self,
        context: MiddlewareContext,
        result: Any
    ) -> Any:
        """ Process after the operation completes successfull. """
        pass    

    @abstractmethod
    async def process_error(
        self,
        context: MiddlewareContext,
        error: Exception
    ) -> Optional[Any]:
        """ Handle errors from the execution. """
        pass

class SecurityMiddleware(BaseMiddlware):
    """ Blocks malicious input before it reaches the input. """

    def __init__(self):
        self.malicious_patterns = [
            r"ignore.*previous.*instructions",
            r"system.*prompt.*injection",
            r"\\x[0-9a-f]{2}", # Hex encoding attempts
            r"<script.*?>.*?</script>", # Script injection
        ]

    async def process_request(
            self,
            context: MiddlewareContext
    ) -> MiddlewareContext:
        """ Block malicious requests befoe they reach the model. """
        if context.operation == "model_call":
            for message in context.data:
                if hasattr(message, "content"):
                    for pattern in self.malicious_patterns:
                        if re.search(
                            pattern,
                            message.content,
                            re.IGNORECASE
                        ):
                            raise ValueError(
                                "Blocked potentially malicious input"
                            )
                        
class LoggingMiddleware(BaseMiddleware):
    """ Log all agent operations before/after hooks. """

    async def process_request(
            self,
            context: MiddlewareContext
    ) -> MiddlewareContext:
        """ Log operation start. """
        print(f"[{context.agent_name}] Starting {context.operation}")
        context.metadata["start_time"] = time.time()
        return context
    
    async def process_response(
            self,
            context: MiddlewareContext,
            result: Any
    ) -> Any:
        """ Log successful operation. """
        duration = time.time() - context.metadata.get("start_time", 0)
        print(f"[{context.agent_name}] {context.operation} completed in {duration: .2f}s")
        return result 
    
    async def process_error(self, context: MiddlewareContext, error: Exception) -> Optional[Any]:
        """ Log operation failure. """
        print(f"[{context.agent_name}] {context.operation} failed {error}")
        raise error 
