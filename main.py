from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import get_tools
import json
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Define expected output format
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7
)

# Parser for structured agent response
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Define agent prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research summary.
            Use tools when necessary and return your response in this structured format:
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Load tools from tools.py
tools = get_tools()

# Create and configure agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Ask user for input
query = input("🔍 What would you like to research? ")

# Run agent with the query
raw_response = agent_executor.invoke({"query": query})

# Parse and display result
try:
    raw_output = raw_response.get("output", "")
    cleaned_output = (
        raw_output.replace("```json", "")
        .replace("```", "")
        .strip()
    )
    structured_response = parser.parse(cleaned_output)
    print("\n✅ Research Result:\n")
    print(structured_response)

except Exception as e:
    print("\n❌ Error parsing response:", e)
    print("📦 Raw Response:\n", raw_response)



