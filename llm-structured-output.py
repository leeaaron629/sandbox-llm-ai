from dotenv import load_dotenv
import os

load_dotenv()

import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage

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

print('Type quit or exit to stop.')
while True:
    
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    response = chat.invoke([HumanMessage(content=user_input)])
    print("Agent:", response.content)
