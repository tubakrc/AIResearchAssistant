# AI Research Assistant

🔎🔗 [https://ai-research-assistant25.streamlit.app/]

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

<img width="1918" height="1020" alt="image" src="https://github.com/user-attachments/assets/394a2ef3-5075-43b2-adcc-2971c82a104f" />

---

## 🛠️ Tech Stack

- **LLM**: Gemini 1.5 Flash via `langchain-google-genai`
- **Search Tools**: DuckDuckGo + Wikipedia (`langchain_community`)
- **Agent Framework**: LangChain `ToolCallingAgent` & `AgentExecutor`
- **Frontend**: Streamlit
- **Output Format**: `Pydantic` for structured data



