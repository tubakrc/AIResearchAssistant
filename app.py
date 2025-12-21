import streamlit as st
from dotenv import load_dotenv
from tools import get_tools, save_to_txt
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langchain.output_parsers import PydanticOutputParser


from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool



import logging
import os


import logging
import os

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

tools = get_tools()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.5
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

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

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

st.set_page_config(page_title="AI Research Assistant", layout="centered")
st.title("📚 AI Research Assistant")

query = st.text_input("🔍 What would you like to research?", placeholder="e.g. Effects of AI on Education")

if st.button("Run Agent"):
    if not query:
        st.warning("Please enter a query before running.")
    else:
        with st.spinner("Thinking... 🧠🧠🧠"):
            try:
                raw_response = agent_executor.invoke({"query": query})
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

                # Dosyayı kaydet (manuel)
                save_message = save_to_txt(structured_response)
                st.success(save_message)

                # Dosya indirme butonu
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
                st.error("❌ Failed to parse response.")
                st.exception(e)















