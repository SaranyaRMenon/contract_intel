# 🚀 AI-Powered Contract Intelligence System

An end-to-end **contract analysis platform** demonstrating advanced AI concepts including **Retrieval-Augmented Generation (RAG), vector databases, structured outputs, and agentic workflows**.

---

## 📌 Overview

This system allows users to upload legal contracts and perform intelligent analysis using AI. It processes documents, builds a semantic search index, and provides grounded insights such as clause extraction, risk detection, and Q&A — all strictly based on contract evidence.

---

## ✨ Key Features

* 📂 **Document Ingestion**
  Supports PDF, DOCX, and TXT files with automated parsing and chunking.

* 🔍 **Semantic Search (RAG)**
  Uses vector embeddings to retrieve relevant contract context for accurate answers.

* 📑 **Clause Extraction**
  Extracts structured clauses using **Pydantic schemas** with validation guardrails.

* ⚠️ **Risk Detection**
  Identifies potential risks using a hybrid **rule-based + LLM approach**.

* 💬 **AI Chatbot**
  Answers contract-related queries with **grounded responses (no hallucination)**.

* 🧠 **Agentic Workflow**
  Intelligent agent decides which tools (search, extract, risk) to use dynamically.

* 📊 **Summarization**
  Generates concise executive summaries tailored for decision-makers.


## 🛠️ Tech Stack

* **Backend:** Python, FastAPI
* **Frontend:** Streamlit
* **AI/ML:** OpenAI (via proxy), LangChain
* **Vector DB:** ChromaDB
* **Validation:** Pydantic



⚙️ **Run Locally**
pip install -r requirements.txt
streamlit run app.py


---


AI & Full Stack Developer
