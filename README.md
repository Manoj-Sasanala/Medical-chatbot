# Medical RAG Chatbot

A Medical AI Chatbot built using Flask, LangChain concepts, Pinecone Vector Database, HuggingFace Embeddings, and DeepSeek LLM API.

## Features

- Medical question answering
- PDF-based knowledge retrieval
- Semantic search using vector embeddings
- Conversational chatbot UI
- Retrieval-Augmented Generation (RAG)
- Pinecone vector database integration
- DeepSeek API integration

---

## Tech Stack

- Python
- Flask
- Pinecone
- HuggingFace Embeddings
- DeepSeek API
- LangChain
- HTML/CSS

---

## Project Architecture

User Question
↓
Embedding Generation
↓
Pinecone Semantic Search
↓
Relevant Context Retrieval
↓
LLM Response Generation
↓
Chatbot Response

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Manoj-Sasanala/Medical-chatbot
cd Medical-chatbot
```
## Dataset

Place your medical PDF inside the `data/` folder before running the project.

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment (Windows)

```bash
.\venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Setup Environment Variables

Create a `.env` file in the root directory and add your API keys:

```env
PINECONE_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
```

### 6. Run Project

```bash
python app.py
```