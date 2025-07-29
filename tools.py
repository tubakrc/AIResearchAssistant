from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
from typing import Optional
import os
import logging

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

import os
from datetime import datetime
import shutil

def save_to_txt(data, filename: str = "research_output.txt") -> str:
    filepath = os.path.join(RESULTS_DIR, filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Önce eski dosya varsa yedekle
    if os.path.exists(filepath):
        backup_name = f"research_output_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        backup_path = os.path.join(RESULTS_DIR, backup_name)
        shutil.copy2(filepath, backup_path)

    # Dosya içeriğini hazırla
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

    # Yeni içeriği dosyaya yaz (w modu ile, eskiyi siler)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        return f"✅ Data successfully saved! "
    except Exception as e:
        return f"❌ Failed to save data: {e}"


def save_to_txt_wrapper(data: str) -> str:
    return save_to_txt(data, "research_output.txt")

def get_tools():
    search = DuckDuckGoSearchRun()
    search_tool = Tool(
        name="search",
        func=search.run,
        description="Search the web for up-to-date information using DuckDuckGo.",
    )

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
    wiki_tool = Tool(
        name="wikipedia",
        func=WikipediaQueryRun(api_wrapper=api_wrapper).run,
        description="Search concise summaries from Wikipedia.",
    )

    save_tool = Tool(
        name="save_text_to_file",
        func=save_to_txt_wrapper,
        description="Saves research data to a timestamped text file.",
    )

    return [search_tool, wiki_tool, save_tool]
