import requests
from langchain.llms.base import LLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from typing import Optional, List, Mapping, Any

llm_agent_task = """
You are a database admin with years of experience writing SQL.

You have access to the following tools:

Question: Ask the user a question to clarify on what they want in their SQL query, such as table name, column names, etc.
Response to Human: Provide a SQL query that answers the user's question

You will receive a message from the human, then you should start a loop and do one of the two things:

Option 1: You use a tool to answer the question.
For this, you should use the following format:
Thought: You should always think about type of SQL query to provide and what table and columns are involved
Action: The action to take, should be one of the tools listed above.
Action Input: The arguments to the tool, should be a valid SQL query

Observation: The result of the action, should be a valid SQL query

Option 2: You response to the human
For this you should use the following format:
Action: Response to Human
Action Input: Action Input: "Your response to the human, summarizing what you understood from the question and the query you generated"

"""

class OllamaLLM(LLM):

    model_name: str = "deepseek-r1:14b"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model_name, "prompt": prompt}
        )
        return response.json()['response']
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}
    
    @property
    def _llm_type(self) -> str:
        return "custom_ollama_llm"

# Initialize the Ollama LLM
ollama_llm = OllamaLLM()

# Create a ConversationChain (our agent) with memory
conversation = ConversationChain(
    llm=ollama_llm,
    memory=ConversationBufferMemory()
)

# Function to interact with the agent
def interact_with_agent(task):
    print(f"Current task: {task}")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        response = conversation.predict(input=f"Task: {task}\nHuman: {user_input}")
        print("Agent:", response)

# Example usage
task = "Remember and execute tasks related to Python programming"
interact_with_agent(task)