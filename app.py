import streamlit as st
from dotenv import load_dotenv
from tools import get_tools, save_to_txt
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate

import logging
import os
import warnings
import json

warnings.filterwarnings("ignore")

# ---------------- UI ----------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #4facfe, #8e44ad);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.set_page_config(page_title="AI Research Assistant", layout="centered")

# ---------------- Setup ----------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FILENAME = "research_output.txt"
FILEPATH = os.path.join(RESULTS_DIR, FILENAME)

# ---------------- Schema ----------------
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# ---------------- Tools ----------------
tools = get_tools()
if not tools:
    st.error("No tools loaded")
    st.stop()

tool_descriptions = "\n".join(
    f"{t.name}: {t.description}" for t in tools
)
tool_names = ", ".join(t.name for t in tools)

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.5
)

# ---------------- Prompt (CRITICAL FIX) ----------------
from langchain_core.prompts import ChatPromptTemplate

template = """Answer the following question as best you can.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: reasoning
Action: the action to take, should be one of [{tool_names}]
Action Input: input to the action
Observation: result of the action
... (repeat as needed)
Thought: I now know the final answer
Final Answer: Return your response in this JSON format:
{format_instructions}

Begin!

Question: {input}
{agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_template(template).partial(
    format_instructions=parser.get_format_instructions()
)

# ---------------- Agent ----------------
try:
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )

except Exception as e:
    st.error("Failed to create agent")
    st.exception(e)
    st.stop()

# ---------------- UI ----------------
st.title("📚 AI Research Assistant")

with st.sidebar:
    st.subheader("Available Tools")
    for t in tools:
        st.write(f"• {t.name}")

query = st.text_input(
    "What would you like to research?",
    placeholder="e.g. Effects of AI on Education",
)

# ---------------- Run ----------------
if st.button("Run Agent"):
    if not query:
        st.warning("Enter a query")
    else:
        with st.spinner("Researching..."):
            try:
                response = agent_executor.invoke({"input": query})

                raw_output = (
                    response.get("output")
                    or response.get("final_answer")
                    or ""
                )

                # strip markdown + guard JSON
                cleaned = (
                    raw_output.replace("```json", "")
                    .replace("```", "")
                    .strip()
                )
                cleaned = cleaned[
                    cleaned.find("{") : cleaned.rfind("}") + 1
                ]

                structured = parser.parse(cleaned)

                st.subheader("Topic")
                st.write(structured.topic)

                st.subheader("Summary")
                st.markdown(structured.summary.replace("\n", "  \n"))

                st.subheader("Sources")
                for s in structured.sources:
                    st.markdown(f"- {s}")

                st.subheader("Tools Used")
                st.write(", ".join(structured.tools_used))

                msg = save_to_txt(structured)
                st.success(msg)

                if os.path.exists(FILEPATH):
                    with open(FILEPATH, "rb") as f:
                        st.download_button(
                            "Download Result",
                            f.read(),
                            file_name=FILENAME,
                            mime="text/plain",
                        )

            except Exception as e:
                st.error("Agent execution failed")
                st.exception(e)

