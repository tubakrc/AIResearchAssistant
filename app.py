import streamlit as st
from dotenv import load_dotenv
from tools import get_tools, save_to_txt
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent

import logging
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', message="Key 'title' is not supported in schema")

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

# Initialize page config first
st.set_page_config(page_title="AI Research Assistant", layout="centered")

# Get tools
tools = get_tools()

# Verify tools loaded
if not tools:
    st.error("⚠️ No tools loaded! Check your tools.py file.")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.5
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Fixed prompt - removed chat_history placeholder which was causing the issue
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a research assistant that helps generate comprehensive research summaries.

Use the available tools to gather information:
- Use 'web_search' or 'duckduckgo_search' for current events, news, and general web information
- Use 'wikipedia' for encyclopedic and historical information

After gathering information, provide your response in this JSON format:
{format_instructions}

Make sure to:
1. Provide a clear, descriptive topic
2. Write a comprehensive summary (3-5 paragraphs)
3. List all sources you used (URLs or Wikipedia pages)
4. List which tools you used
""",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

try:
    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=tools,
    )

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
except Exception as e:
    st.error(f"❌ Failed to create agent: {str(e)}")
    st.exception(e)
    st.stop()

st.title("📚 AI Research Assistant")

# Show loaded tools in sidebar
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
                # Invoke agent - use 'input' instead of 'query'
                raw_response = agent_executor.invoke({"input": query})
                raw_output = raw_response.get("output", "")
                
                # Clean and parse output
                cleaned_output = (
                    raw_output.replace("```json", "")
                    .replace("```", "")
                    .strip()
                )
                
                structured_response = parser.parse(cleaned_output)

                # Display results
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

                # Save to file
                save_message = save_to_txt(structured_response)
                st.success(save_message)

                # Download button
                if os.path.exists(FILEPATH):
                    with open(FILEPATH, "rb") as f:
                        file_bytes = f.read()
                    st.download_button(
                        label="⬇️ Download Latest Research File",
                        data=file_bytes,
                        file_name=FILENAME,
                        mime="text/plain"
                    )
                else:
                    st.info("ℹ️ File not found yet or not saved.")

            except Exception as e:
                st.error("❌ Failed to complete research.")
                st.exception(e)
                
                # Show raw output for debugging
                with st.expander("🔍 Debug Info"):
                    st.write("Raw response:", raw_response if 'raw_response' in locals() else "No response")
                    if 'raw_output' in locals():
                        st.write("Raw output:", raw_output)
