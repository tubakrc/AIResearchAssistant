import streamlit as st
from dotenv import load_dotenv
from tools import get_tools, save_to_txt
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

import logging
import os
import warnings

warnings.filterwarnings('ignore')

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #4facfe, #8e44ad);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

load_dotenv()
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FILENAME = "research_output.txt"
FILEPATH = os.path.join(RESULTS_DIR, FILENAME)

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

st.set_page_config(page_title="AI Research Assistant", layout="centered")

# Get tools
tools = get_tools()

if not tools:
    st.error("⚠️ No tools loaded!")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.5
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Use ReAct agent instead of tool_calling_agent
template = '''Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: Return your response in this JSON format:
{format_instructions}

Begin!

Question: {input}
Thought: {agent_scratchpad}'''

prompt = PromptTemplate.from_template(template).partial(
    format_instructions=parser.get_format_instructions()
)

try:
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
except Exception as e:
    st.error(f"❌ Failed to create agent: {str(e)}")
    st.exception(e)
    st.stop()

st.title("📚 AI Research Assistant")

with st.sidebar:
    st.subheader("🛠️ Available Tools")
    for tool in tools:
        tool_name = tool.name if hasattr(tool, 'name') else str(type(tool).__name__)
        st.write(f"✅ {tool_name}")

query = st.text_input("🔍 What would you like to research?", placeholder="e.g. Effects of AI on Education")

if st.button("Run Agent"):
    if not query:
        st.warning("Please enter a query before running.")
    else:
        with st.spinner("Researching... 🧠🧠🧠"):
            try:
                raw_response = agent_executor.invoke({"input": query})
                raw_output = raw_response.get("output", "")
                
                cleaned_output = (
                    raw_output.replace("```json", "")
                    .replace("```", "")
                    .strip()
                )
                
                structured_response = parser.parse(cleaned_output)

                st.subheader("📝 Topic")
                st.markdown(f"**{structured_response.topic}**")

                st.subheader("📄 Summary")
                st.markdown(structured_response.summary.replace("\n", "  \n"))

                st.subheader("🔗 Sources")
                for s in structured_response.sources:
                    if s.startswith("http"):
                        st.markdown(f"- [{s}]({s})")
                    else:
                        st.markdown(f"- {s}")

                st.subheader("🛠️ Tools Used")
                st.markdown(", ".join(structured_response.tools_used))

                save_message = save_to_txt(structured_response)
                st.success(save_message)

                if os.path.exists(FILEPATH):
                    with open(FILEPATH, "rb") as f:
                        file_bytes = f.read()
                    st.download_button(
                        label="⬇️ Download Latest Research File",
                        data=file_bytes,
                        file_name=FILENAME,
                        mime="text/plain"
                    )

            except Exception as e:
                st.error("❌ Failed to complete research.")
                st.exception(e)
                
                with st.expander("🔍 Debug Info"):
                    if 'raw_response' in locals():
                        st.write("Raw response:", raw_response)
                    if 'raw_output' in locals():
                        st.write("Raw output:", raw_output)
