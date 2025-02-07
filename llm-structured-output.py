from dotenv import load_dotenv
import os

load_dotenv()

import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver

vertexai.init(project="tailwind-370000")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
chat = ChatVertexAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)

# messages = [
#     SystemMessage(content="You're a helpful assistant"),
#     HumanMessage(content="What is the purpose of model regularization?"),
# ]

# chat.invoke(messages)

# for chunk in chat.stream(messages):
#     print(chunk.content, end="", flush=True)

class Task(BaseModel):
    task_name: str = Field(description="The name of the task")
    task_description: str = Field(description="The description of the task")

    def get_content(self):
        return f"Setup: {self.setup}\nPunchline: {self.punchline}"

structured_llm = chat.with_structured_output(Task)    

print('Give me a list of tasks and their descriptions')
print('Type quit or exit to stop.')
AGENT_TASK = """
You are a personal assistant for a user. You will be given a list of tasks and their descriptions.

Then you will respond to the user with your understanding of the tasks and their descriptions and ask for a confirmation.
If they confirm, you will reply back with: "Ok noting down tasks and their descriptions".
If they don't confirm, you will ask for further clarifications and then reply back with your current understanding of the tasks and their descriptions.
Repeat this process until the user confirms.
"""
# chat.invoke([SystemMessage(content=AGENT_TASK)])
while True:
    
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    response = chat.invoke([HumanMessage(content=user_input)])
    print("Agent:", response.content)
