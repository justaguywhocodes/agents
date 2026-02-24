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

    def _process_tools(
            self,
            tools: List[Union[BaseTool, Callable]]
    ) -> List[BaseTool]:
        """ Convert mixed tool types to BaseTool instances """
        processed = []

        for tool in tools:
            if isinstance(tool, BaseTool):
                processed.apend(tool)
            elif callable(tool):
                # Wrap functions in FunctionTool automatically
                processed.append(FunctionTool(tool))
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}")
        return processed 
    
    def _get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """ Convert tools to LLM function calling format """
        return [tool._to_llm_format() for tool in self.tools]
    
    async def _execute_tool_call(
        self,
        tool_call: ToolCallRequest
    ) -> ToolMessage:
        """ Execute a tool call and return result message """
        # Find the requested tool by name
        tool = self._find_tool(tool_call.tool_name)
        if tool is None: 
            return ToolMessage(
                content=f"Tool {tool_call.tool_name} not found",
                tool_call_id=tool_call.call_id,
                success=False,
                error=f"Tool not found"
            )
        
        # Execute the tool with error handling
        try:
            result = await tool.execute(tool_call.parameters)
            content = (
                str(result.result) if result.success
                else f"Error: {result.error}"
            )
            return ToolMessage(
                content=content,
                tool_call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=result.success,
                error=result.error
            )
        except Exception as e:
            # Handle unexpected errors gracefully
            return ToolMessage(
                content=f"Tool call execution failed: {str(e)}",
                tool_call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=False,
                error=str(e)
            )
        
    async def _prepare_llm_messages(
            self,
            task_messages: List[Message],             
    ) -> List[Message]:
        """Prepare context with memory integration."""
        system_context = self.instructions

        # Memory provides relevant context through get_context()
        if self.memory:
            context = await self.memory.get_context(max_items=5)
            if context:
                system_context += f"\n\nRelevant context: \n{'\n'.join(context)}"
        
        return [
            SystemMessage(content=system_content),
            *self.message_history,
            *task_messages
        ]