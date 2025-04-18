# 📈 AI Financial Research Assistant

> Your AI-powered assistant for real-time stock insights, analyst recommendations, company news, and web intelligence — all in one chat interface.

![Screenshot](./assets/screenshot.png) <!-- Optional: Replace with your own screenshot path -->

---

## 🚀 Overview

The **AI Financial Research Assistant** is a multi-agent LLM-powered web application built with Streamlit and `phi`. It allows users to interact in natural language and get real-time insights into the stock market — including prices, fundamentals, analyst opinions, and recent news.

Perfect for investors, researchers, and finance students seeking clear, trustworthy, and fast financial information.

---

## ✨ Features

- 🤖 **Multi-Agent LLM architecture** using [phi](https://github.com/opscientia/phi)
  - **Finance Agent**: Stock prices, fundamentals, analyst recommendations
  - **Web Search Agent**: Latest news and updates via DuckDuckGo
- 🚀 Powered by **Groq** with **DeepSeek LLaMA 70B** for ultra-fast reasoning
- 💬 **Natural language chat interface** via Streamlit
- 📊 Stock visuals and clean data tables
- 🛠️ Caching + retry handling to manage rate limits
- 🔐 `.env` + `streamlit secrets` support for API keys

---

## 📸 Demo

Example queries you can try:

- "Summarize analyst recommendation and latest news for **Nvidia (NVDA)**"
- "What's the **current price and P/E ratio** of **Apple (AAPL)**?"
- "Compare **Microsoft and Google** stock performance"
- "Find recent headlines about **Tesla**"

---

## 🧠 Architecture

---

## 🛠️ Tech Stack

| Tool            | Purpose                                  |
|-----------------|------------------------------------------|
| `Streamlit`     | Frontend Chat Interface                  |
| `phi`           | Multi-agent Framework                    |
| `Groq`          | LLM Inference (DeepSeek LLaMA 70B)       |
| `YFinanceTools` | Stock data, analyst opinions, fundamentals |
| `DuckDuckGo`    | Real-time web search                     |
| `dotenv`        | Environment variable management          |

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/your-username/ai-financial-assistant.git
cd ai-financial-assistant

# Install dependencies

# Add your API key in .env
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# Run the app
streamlit run app.py

