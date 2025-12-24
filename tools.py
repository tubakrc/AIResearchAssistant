from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from datetime import datetime
import os
import logging
import shutil

logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_to_txt(data, filename: str = "research_output.txt") -> str:
    filepath = os.path.join(RESULTS_DIR, filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save backup if there is an old file
    if os.path.exists(filepath):
        backup_name = f"research_output_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        backup_path = os.path.join(RESULTS_DIR, backup_name)
        shutil.copy2(filepath, backup_path)
    
    # Prepare the file content
    try:
        formatted_text = (
            f"--- Research Output ---\n"
            f"Timestamp: {timestamp}\n\n"
            f"Topic: {data.topic}\n\n"
            f"Summary:\n{data.summary}\n\n"
            f"Sources:\n" + "\n".join(f"- {src}" for src in data.sources) + "\n\n"
            f"Tools Used:\n" + ", ".join(data.tools_used) + "\n\n"
            "-----------------------------------------\n\n"
        )
    except AttributeError:
        formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{str(data)}\n\n"
    
    # Write the new content to the file
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        return f"✅ Data successfully saved!"
    except Exception as e:
        return f"❌ Failed to save data: {e}"

def get_tools():
    """Return a list of tools compatible with create_tool_calling_agent"""
    
    # DuckDuckGo search tool - use directly without Tool wrapper
    search_tool = DuckDuckGoSearchRun(
        name="web_search",
        description="Search the web for up-to-date information using DuckDuckGo. Use this for current events, news, and general web searches."
    )
    
    # Wikipedia tool - use directly without Tool wrapper
    api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
    wiki_tool = WikipediaQueryRun(
        api_wrapper=api_wrapper,
        name="wikipedia",
        description="Search Wikipedia for encyclopedic information. Use this for historical facts, biographies, and general knowledge."
    )
    
    return [search_tool, wiki_tool]
