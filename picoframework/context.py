# context.py
from typing import List
from messages import Message, UserMessage, AssistantMessage, SystemMessage

class AgentContext:
    def __init__(self):
        self.messages = List[Message] = []

        def add_message(self, message: Message):
            self.messages.append(message)
            # Optionally truncate to keep recent context
            if len(self.messages) > 50
            self.messages = self.messages[-50:]
