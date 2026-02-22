from typing import AsyncGenerator, List, Optional, Union
from base_agent import BaseAgent
from messages import SystemMessage, UserMessage, AssistantMessage, ToolMessage, Message
from context import AgentContext

class Agent(BaseAgent):
    async def run_stream(
            self,
            task: Union[str, UserMessage, List[Message]]
    ) -> AsyncGenerator[Union[Message, AgentEvent], None]:
        # prepare context with instructions and history
        llm_messages = [
            SystemMessage(content=self.instructions),
            *self.context.messages,
            *task_messages
        ]
        # call model client 
        completion_result = await self.model_client.create(llm_messages)
        assistant_message = completion_result.message

        # Yield the response 
        yield assistant_message

        # update conversation context
        self.context.add_message(assistant_message)