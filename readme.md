# 🧠 RAG PDF Agent — Local LLM + Visual Embedding Explorer

<p align="center">
  <a href="https://ollama.com/">
    <img src="https://img.shields.io/badge/Ollama-Local_LLM_Engine-green?style=for-the-badge&logo=OpenAI&logoColor=white" alt="Ollama LLM Engine"/>
  </a>
  <a href="https://gradio.app/">
    <img src="https://img.shields.io/badge/Gradio-UI_Framework-blue?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio UI"/>
  </a>
</p>

This repository contains a modular, lightweight RAG (Retrieval-Augmented Generation) pipeline that allows uploading **multiple PDFs**, builds an on-the-fly **vector database**, and queries them using **local open-source LLMs** (via Ollama). It also supports **2D embedding visualizations** using UMAP + Matplotlib.

This project is part of my **SSJ3-AI-Agent Journey**, demonstrating modular, local-first, low-resource-friendly AI applications.

---

## 🎯 Objective

- 🧾 Upload multiple PDFs (papers, manuals, docs)
- 📚 Build per-session vector DBs (no persistence)
- 🧠 Use local LLMs (like `phi`, `mistral`) via **Ollama**
- 🔎 Ask questions and get answers based on document context
- 📊 Visualize embedding clusters using **UMAP**

---

## 📁 Features

| Component | Tech Stack |
|----------|------------|
| PDF Reading | `PyMuPDF` |
| Chunking | `LangChain`'s `RecursiveCharacterTextSplitter` |
| Embedding | `sentence-transformers` (`MiniLM`) |
| Vector DB | `FAISS` |
| LLM | `phi` / `mistral` via `Ollama` |
| UI | `Gradio` |
| Visualization | `Matplotlib` + `UMAP` |

---

## 🧭 Workflow

### ✅ RAG Pipeline

| Step | Description |
|------|-------------|
| 📤 Upload PDFs | Upload multiple documents |
| 🧠 Chunk & Embed | Split content into chunks and generate embeddings |
| 🗃️ Vector DB | Store embeddings using FAISS |
| 🔍 Ask Question | Top-K retrieval from DB |
| 🧾 Prompt LLM | Create prompt using retrieved context |
| 🤖 Answer | Get response from local LLM via Ollama |
| 📈 Visualize | Show 2D embedding clusters using UMAP |

---

## 🧰 Installation Guide

### ✅ Python Environment

```bash
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
```

### ✅ Install Python Dependencies

```bash
pip install gradio pymupdf faiss-cpu sentence-transformers matplotlib umap-learn
```

### ✅ Install Ollama
- Download and install from: https://ollama.com/download
- Pull a small model like Phi or Mistral:

```bash
ollama pull phi
```
- ℹ️ Your system should be able to run these with 8 GB RAM + CPU setup (no GPU required).

## 🚀 How to Run

```bash
python rag_gradio_app.py
```

---

## 🚀 How to Use

After running the app:

1. 📤 **Upload PDFs** (you can select multiple files)  
2. 📁 Click **"Create Vector DB"** to build a new vector store  
3. ❓ Ask any question about the uploaded documents  
4. 📊 View the **2D Embedding Map** generated using UMAP + Matplotlib

---

## 📊 Visualizations

| Feature               | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| 🧬 UMAP Embedding      | Clusters chunks based on semantic similarity                  |
| 📈 2D Plot             | Saved dynamically and displayed in the UI                     |
| 🌐 Interactive Explore | Can be enhanced later using `plotly` or `altair`              |
| ☁️ Word Cloud          | Highlights frequent keywords relevant for queries             |
| 🖼️ UI Screenshots      | Interface view of deployed Gradio application                |

<p align="center"> 
  <img src="wordcloud_plot.png" alt="WordCloud Visualization" width="600"/> 
</p>

<p align="center"> 
  <img src="deployed_app_img1.png" alt="Deployed App Screenshot 1" width="600"/>
</p>

<p align="center"> 
  <img src="deployed_app_img2.png" alt="Deployed App Screenshot 2" width="600"/>
</p>
---

## 📦 Folder Structure

```bash
rag-pdf-agent-visual/
├── rag_gradio_app.py           # Main app file
├── requirements.txt            # All required packages
├── README.md                   # This file
├── /tmp/embedding_plot_*.png   # Saved embedding visualizations (auto)
```

---

## 🧾 Sample Usage

```bash
Q: What is the main finding in the uploaded research paper?
A: Based on the context, the authors propose a lightweight CNN for edge devices with 90.3% accuracy...

Q: Summarize the instruction manual.
A: The document outlines 3 core steps: Setup, Calibration, and Testing. Safety is emphasized.
```

---

## 🧠 Future Work

- [ ] Save/load named vector DBs for reuse  
- [ ] Add file-level metadata context  
- [ ] Use `LlamaIndex` as an alternative engine  
- [ ] Add PDF preview and context window in the UI  
- [ ] Deploy on local server with user authentication  

---

## 📚 Learnings

- Efficient RAG pipelines are achievable with low-end hardware  
- FAISS and UMAP offer fast, visual insights into text similarity  
- Modular class-based design allows extensibility  
- Ollama makes running local LLMs extremely easy  

---

## ✍️ Author

- **Name**: Guna Venkat Doddi  
- **Project**: Part of `SSJ3-AI-Agent-Projects`  
- **Contact**: [![GitHub - Guna Venkat Doddi](https://img.shields.io/badge/GitHub-Guna--Venkat--Doddi-black?logo=github&style=flat-square)](https://github.com/Guna-Venkat)
