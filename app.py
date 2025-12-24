import streamlit as st
from dotenv import load_dotenv
from tools import get_tools, save_to_txt
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import PromptTemplate

import logging
import os
import warnings

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
format_instructions = parser.get_format_instructions()

# ---------------- Tools ----------------
tools = get_tools()
if not tools:
    st.error("No tools loaded")
    st.stop()

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.5,
)

# ---------------- ReAct Prompt (JSON ENFORCED) ----------------
REACT_TEMPLATE = """
Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: think step by step
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this can repeat)
Thought: I now know the final answer
Final Answer: MUST be valid JSON only

IMPORTANT:
Your Final Answer MUST strictly follow this JSON schema:
{format_instructions}

Do NOT include markdown, explanations, or text outside JSON.

Begin!

Question: {input}
{agent_scratchpad}
"""


prompt = PromptTemplate(
    template=REACT_TEMPLATE,
    input_variables=[
        "input",
        "agent_scratchpad",
        "tools",
        "tool_names",
    ],
).partial(
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
        max_iterations=15,
        early_stopping_method="generate"
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

                cleaned = (
                    raw_output.replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

                # -------- SAFE PARSING --------
                try:
                    structured = parser.parse(cleaned)
                except Exception:
                    structured = ResearchResponse(
                        topic=query,
                        summary=raw_output,
                        sources=[],
                        tools_used=[t.name for t in tools],
                    )

                # -------- UI OUTPUT --------
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



