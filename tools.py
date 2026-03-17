from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from datetime import datetime
from pydantic import BaseModel, Field
import os, logging, shutil, time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_to_txt(data, filename: str = "research_output.txt") -> str:
    filepath = os.path.join(RESULTS_DIR, filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(filepath):
        backup_name = f"research_output_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        shutil.copy2(filepath, os.path.join(RESULTS_DIR, backup_name))

    try:
        formatted_text = (
            f"--- Research Output ---\n"
            f"Timestamp: {timestamp}\n\n"
            f"Topic: {data.topic}\n\n"
            f"Summary:\n{data.summary}\n\n"
            f"Sources:\n" + "\n".join(f"- {src}" for src in data.sources) + "\n\n"
            f"Tools Used:\n" + ", ".join(data.tools_used) + "\n"
            f"-----------------------------------------\n"
        )
    except AttributeError:
        formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{str(data)}\n"

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        return "✅ Data successfully saved!"
    except Exception as e:
        return f"❌ Failed to save data: {e}"


class RobustDuckDuckGoSearch(DuckDuckGoSearchRun):
    """DuckDuckGo with retry logic to handle rate limiting."""

    def _run(self, query: str) -> str:
        max_retries = 3
        retry_delay = 2.0

        for attempt in range(max_retries):
            try:
                result = super()._run(query)
                if result and result.strip():
                    return result
                logger.warning(f"DDG returned empty result on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"DDG attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
        return "DuckDuckGo search temporarily unavailable. Wikipedia results will be used."


def get_tools():
    ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
    search_tool = RobustDuckDuckGoSearch(api_wrapper=ddg_wrapper)

    wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=800)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    return [search_tool, wiki_tool]
