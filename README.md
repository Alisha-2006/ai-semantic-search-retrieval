<div align="center">

# AI-Powered Semantic Search and Information Retrieval

### Intelligent semantic memory and change detection using AI

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-orange)](https://www.langchain.com/langgraph)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green?logo=mongodb)](https://www.mongodb.com/)
[![Gemini](https://img.shields.io/badge/Google-Gemini-blue)](https://ai.google.dev/)
[![MiniLM](https://img.shields.io/badge/Embeddings-MiniLM-purple)](https://www.sbert.net/)

</div>

---

## 📌 Project Purpose

The project is designed to **store and retrieve information based on meaning rather than exact keywords**. It uses semantic embeddings and AI-based validation to identify related information and detect meaningful changes over time.

---

## ✨ Features

*  AI-based fact extraction from unstructured text
*  Dynamic fact categorization
*  Semantic search using vector embeddings
*  384-dimensional MiniLM embeddings
*  MongoDB Vector Search
*  Gemini-based same-fact validation
*  Automatic change detection
*  Duplicate memory prevention
*  LangGraph workflow orchestration

---

## 🔄 How It Works

```text
┌─────────────────────┐
│   Input Information │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Gemini Fact         │
│ Extraction          │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Structured Facts    │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ MiniLM Embeddings   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ MongoDB Vector      │
│ Search              │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Same-Fact Validation│
│      with Gemini    │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Value Comparison    │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Change Detection    │
└─────────────────────┘
```

### 🧠 Fact Extraction

Gemini converts important information into structured facts:

```text
Type
Topic
Value
Source Text
```

Example:

```text
Type: technology
Topic: mobile application development
Value: React Native
```

The system dynamically identifies the type and topic instead of using a fixed category list.

### 🔎 Semantic Search

Facts are converted into **384-dimensional embeddings** using:

`sentence-transformers/all-MiniLM-L6-v2`

The embeddings are stored in MongoDB and used for semantic retrieval.

### ✅ Same-Fact Validation

Retrieved candidates are validated by **Gemini 3.5 Flash-Lite** to determine whether they represent the same underlying fact.

### 🔄 Change Detection

If the facts refer to the same information, their values are compared.

**Previous:**

```text
Flutter
```

**Current:**

```text
React Native
```

> 🔄 **Change Detected**

Value normalization also helps prevent false changes such as `REST API` vs `REST APIs`.

---

## 🛠️ Technology Stack

| Technology                       | Purpose                        |
| -------------------------------- | ------------------------------ |
| **Python**                       | Main programming language      |
| **LangGraph**                    | Workflow orchestration         |
| **Google Gemini 3.5 Flash-Lite** | Fact extraction and validation |
| **Sentence Transformers**        | Embedding generation           |
| **all-MiniLM-L6-v2**             | 384-dimensional embeddings     |
| **MongoDB Atlas**                | Memory storage                 |
| **MongoDB Vector Search**        | Semantic retrieval             |
| **PyMongo**                      | MongoDB connection             |
| **python-dotenv**                | Environment variables          |

---

## 📁 Project Structure

```text
ai-semantic-search-retrieval/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Alisha-2006/ai-semantic-search-retrieval.git
cd ai-semantic-search-retrieval
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
MONGODB_URI=your_mongodb_connection_string
```

### 4. Run the application

```bash
python app.py
```

---

## 🗄️ MongoDB Configuration

The MongoDB Vector Search index uses:

| Setting        | Value          |
| -------------- | -------------- |
| **Index**      | `vector_index` |
| **Field**      | `embedding`    |
| **Dimensions** | `384`          |
| **Similarity** | `Cosine`       |

---

## 💡 Example

### Previous Information

> The mobile application will be developed using **Flutter**.

### New Information

> The mobile application will be developed using **React Native**.

The system retrieves the previous memory, validates that both statements refer to the same fact, and compares their values.

```text
Previous Value: Flutter
Current Value: React Native

→ Change Detected
```
