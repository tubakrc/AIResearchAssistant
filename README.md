# AI Research Assistant

🔎➡️ [https://ai-research-assistant25.streamlit.app/]

# 🧠 AI Research Assistant – with DuckDuckGo, Wikipedia & Gemini

An AI-powered research agent that automates web + Wikipedia search using tools, summarizes results with Gemini (Google's LLM), and lets users download clean summaries — all through a user-friendly Streamlit interface.

## 🚀 Features

- 🔍 Searches **DuckDuckGo** and **Wikipedia** via LangChain tools  
- 🧑‍🔬 Uses **Gemini 1.5 Flash** to generate structured research summaries  
- 📄 Exports results to a **timestamped text file**  
- 📥 User can download the result as `.txt` directly from the UI   
- 🌐 Clean Streamlit UI for easy interaction  
- 📦 Modular code structure, easy to extend with new tools or agents

---

## 📸 Demo

<img width="1917" height="1018" alt="image" src="https://github.com/user-attachments/assets/c16a3f09-57f4-41ce-96fa-6745dc1c131f" />

---

## 🛠️ Tech Stack

- **LLM**: Gemini 1.5 Flash via `langchain-google-genai`
- **Search Tools**: DuckDuckGo + Wikipedia (`langchain_community`)
- **Agent Framework**: LangChain `ToolCallingAgent` & `AgentExecutor`
- **Frontend**: Streamlit
- **Output Format**: `Pydantic` for structured data



