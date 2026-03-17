import streamlit as st
from dotenv import load_dotenv
from tools import get_tools, save_to_txt
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

import logging, os, warnings

warnings.filterwarnings("ignore")


# ---------------- BOOTSTRAP ----------------
@st.cache_resource
def bootstrap():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

bootstrap()


# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="AI Research Assistant", layout="centered")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #4facfe, #8e44ad);
}
</style>
""", unsafe_allow_html=True)


# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FILENAME = "research_output.txt"
FILEPATH = os.path.join(RESULTS_DIR, FILENAME)


# ---------------- SCHEMA ----------------
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


parser = PydanticOutputParser(pydantic_object=ResearchResponse)


# ---------------- AGENT (tools + executor birlikte cache'lendi) ----------------
@st.cache_resource
def init_agent():
    tools = get_tools()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.5,
    )

    # tool_calling_agent: ReAct'tan çok daha stabil, JSON parse sorunu yok
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research assistant that generates structured research summaries.
Use the available tools to search for information, then return your findings.
Always use both DuckDuckGo and Wikipedia tools when possible.
{format_instructions}"""),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,  # 15→10: gereksiz döngüleri keser
        return_intermediate_steps=False,
    )

    return executor, tools


agent_executor, tools = init_agent()


# ---------------- UI ----------------
st.title("📚 AI Research Assistant")
st.caption("Agent initializes once per session.")

with st.sidebar:
    st.subheader("Available Tools")
    for t in tools:
        st.write(f"• {t.name}")

query = st.text_input(
    "What would you like to research?",
    placeholder="e.g. Effects of AI on Education",
)


# ---------------- RUN ----------------
if st.button("🔍 Run Agent"):
    if not query.strip():
        st.warning("Please enter a research topic.")
    else:
        with st.spinner("🔬 Researching... this may take 20–40 seconds."):
            try:
                response = agent_executor.invoke({"query": query})
                raw_output = response.get("output", "")

                cleaned = (
                    raw_output.replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

                try:
                    structured = parser.parse(cleaned)
                except Exception:
                    structured = ResearchResponse(
                        topic=query,
                        summary=raw_output,
                        sources=[],
                        tools_used=[t.name for t in tools],
                    )

                # -------- DISPLAY --------
                st.subheader("📌 Topic")
                st.write(structured.topic)

                st.subheader("📝 Summary")
                st.markdown(structured.summary.replace("\n", "  \n"))

                if structured.sources:
                    st.subheader("🔗 Sources")
                    for s in structured.sources:
                        st.markdown(f"- {s}")

                st.subheader("🛠️ Tools Used")
                st.write(", ".join(structured.tools_used))

                msg = save_to_txt(structured)
                st.success(msg)

                if os.path.exists(FILEPATH):
                    with open(FILEPATH, "rb") as f:
                        st.download_button(
                            "📥 Download Result",
                            f.read(),
                            file_name=FILENAME,
                            mime="text/plain",
                        )

            except Exception as e:
                st.error("❌ Agent execution failed.")
                st.exception(e)
