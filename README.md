# AI Research Assistant

Check out the app below ➡️➡️➡️

🔗 [https://ai-research-assistant25.streamlit.app/]

## 🧠 AI Research Assistant – with DuckDuckGo, Wikipedia & Gemini

An AI-powered research agent that automates web + Wikipedia search using tools, summarizes results with Gemini (Google's LLM), and lets users download clean summaries — all through a user-friendly Streamlit interface.

## 🚀 Features

🔍 Searches **DuckDuckGo** and **Wikipedia** via LangChain tools  

🧑‍🔬 Uses **Gemini-2.5-flash** to generate structured research summaries  

📄 Exports results to a **timestamped text file**  

📥 User can download the result as `.txt` directly from the UI   

🌐 Clean Streamlit UI for easy interaction  

📦 Modular code structure, easy to extend with new tools or agents

🪿 DuckDuckGo retry logic — rate limit'te 3 trial + exponential backoff

📞 Gemini native function calling — solved JSON parse error

---

## 📸 Demo

<img width="1911" height="2086" alt="AIResearchAssistant" src="https://github.com/user-attachments/assets/959cec9a-b63a-4897-9437-b9eeaef702bf" />

---

## 🛠️ Tech Stack

- **LLM**: Gemini-2.5-flash via `google-generativeai native`
- **Search Tools**: DuckDuckGo + Wikipedia (`langchain_community`)
- **Agent Framework**: Gemini native function calling loop
- **Frontend**: Streamlit
- **Output Format**: `Pydantic` for structured data



