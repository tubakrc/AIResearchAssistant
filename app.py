import streamlit as st
from dotenv import load_dotenv
from tools import get_tools, save_to_txt
from pydantic import BaseModel
import google.generativeai as genai
import json, logging, os, warnings

warnings.filterwarnings("ignore")

# ---------------- BOOTSTRAP ----------------
@st.cache_resource
def bootstrap():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

bootstrap()

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="AI Research Assistant", layout="centered")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #132261;
}
[data-testid="stSidebarContent"] {
    background-color: #FF94AF;
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

# ---------------- TOOL REGISTRY ----------------
@st.cache_resource
def load_tools():
    return {t.name: t for t in get_tools()}

tool_registry = load_tools()

# Gemini function declarations
TOOL_DECLARATIONS = [
    {
        "name": "duckduckgo_search",
        "description": "Search the web using DuckDuckGo. Use for recent news, general facts, and current information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "wikipedia",
        "description": "Search Wikipedia for encyclopedic information about a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic to look up on Wikipedia"}
            },
            "required": ["query"]
        }
    }
]

# ---------------- AGENT RUNNER ----------------
def run_research(query: str) -> ResearchResponse:
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",   
        tools=[{"function_declarations": TOOL_DECLARATIONS}],
        system_instruction="""You are a research assistant.

STEP 1: Use duckduckgo_search tool with the user's query.
STEP 2: Use wikipedia tool with the user's query.
STEP 3: Use duckduckgo_search again with a more specific query if needed.
STEP 4: Synthesize ALL tool results into a detailed summary.

Rules:
- Your summary MUST include specific facts, methods, and details from the tool results.
- Do NOT say tools were unavailable or results were not found if you received any output.
- Minimum summary length: 300 words.
- Always return a JSON object with these exact fields:
{
  "topic": "the research topic",
  "summary": "detailed summary using ALL gathered information",
  "sources": ["list of sources or URLs found"],
  "tools_used": ["duckduckgo_search", "wikipedia"]
}
Return ONLY the JSON. No markdown, no extra text.""")

        

    messages = [{"role": "user", "parts": [query]}]
    tools_used = []
    sources = []

    # Agentic loop — max 6 tur
    for _ in range(6):
        response = model.generate_content(messages)
        candidate = response.candidates[0]
        content = candidate.content

        # Tool call var mı?
        function_calls = [
            p for p in content.parts
            if hasattr(p, "function_call") and p.function_call.name
        ]

        if not function_calls:
            # Final cevap
            text = "".join(
                p.text for p in content.parts if hasattr(p, "text")
            ).strip()
            break

        # Tool'ları çalıştır
        messages.append({"role": "model", "parts": content.parts})
        tool_responses = []

        for part in function_calls:
            fn = part.function_call
            tool_name = fn.name
            args = dict(fn.args)
            tool_query = args.get("query", "")

            tools_used.append(tool_name)
            sources.append(tool_query)

            # LangChain tool'unu çalıştır
            lc_tool = tool_registry.get(tool_name)
            if lc_tool:
                try:
                    result = lc_tool.run(tool_query)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Tool '{tool_name}' not found."

            tool_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={"result": result}
                    )
                )
            )

        messages.append({"role": "user", "parts": tool_responses})

    else:
        text = ""

    # JSON parse
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        return ResearchResponse(**data)
    except Exception:
        return ResearchResponse(
            topic=query,
            summary=text or "Could not parse response.",
            sources=sources,
            tools_used=list(set(tools_used)) or list(tool_registry.keys())
        )

# ---------------- UI ----------------
st.title("📚 AI Research Assistant")
st.caption("Powered by Gemini + DuckDuckGo + Wikipedia")

with st.sidebar:
    st.subheader("Available Tools")
    for name in tool_registry:
        st.write(f"• {name}")

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
                structured = run_research(query)

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
