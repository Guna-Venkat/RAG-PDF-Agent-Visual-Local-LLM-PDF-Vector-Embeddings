import gradio as gr
import fitz
import os
import faiss
import numpy as np
import matplotlib.pyplot as plt
import umap
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import openai
from openai import OpenAI
import ollama as ollama_client
from wordcloud import WordCloud
from collections import Counter
import re

class RAGPipeline:
    def __init__(self, pdf_files, embedding_model="all-MiniLM-L6-v2", model_source="ollama"):
        self.model = SentenceTransformer(embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.pdf_files = pdf_files
        self.model_source = model_source
        self.chunks = []
        self.chunk_id_map = {}
        self.index = None
        self.embeddings = None
        self.build_vector_store()

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def build_vector_store(self):
        all_text = ""
        for file in self.pdf_files:
            all_text += self.extract_text_from_pdf(file.name)

        self.chunks = self.text_splitter.split_text(all_text)
        self.embeddings = self.model.encode(self.chunks)
        dim = len(self.embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings))
        self.chunk_id_map = {i: chunk for i, chunk in enumerate(self.chunks)}

    def retrieve_top_k(self, query, k=5):
        query_vec = self.model.encode([query])
        D, I = self.index.search(np.array(query_vec), k)
        return [self.chunk_id_map[i] for i in I[0]]

    def generate_prompt(self, query):
        context_chunks = self.retrieve_top_k(query)
        context = "\n".join(context_chunks)
        return f"Use the following context to answer the question:\n\n{context}\n\nQ: {query}\nA:"

    def query_llm(self, prompt):
        if self.model_source == "ollama":
            try:
                response = ollama_client.chat(
                    model='phi',
                    messages=[{"role": "user", "content": prompt}]
                )
                return response['message']['content']
            except Exception as e:
                return f"Ollama Error: {str(e)}"

        elif self.model_source == "openai":
            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"OpenAI Error: {str(e)}"

    def visualize_embeddings(self):
        # Join all chunks into one string
        text_corpus = " ".join(self.chunks).lower()

        # Basic cleanup: remove numbers and punctuation
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text_corpus)

        # Tokenize
        words = cleaned_text.split()

        # You can optionally filter out stopwords
        stopwords = set([
            "the", "and", "to", "of", "in", "a", "for", "with", "on", "is", "that", "as", "are", 
            "at", "by", "an", "be", "or", "from", "this", "it", "which", "has", "was", "have"
        ])
        filtered_words = [word for word in words if word not in stopwords and len(word) > 2]

        word_freq = Counter(filtered_words)

        wordcloud = WordCloud(width=800, height=500, background_color='white').generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        file_path = f"wordcloud_plot_{uuid.uuid4().hex}.png"
        plt.savefig(file_path)
        plt.close()
        return file_path

rag_pipeline = None

def create_vector_db(files, model_type):
    global rag_pipeline
    rag_pipeline = RAGPipeline(pdf_files=files, model_source=model_type)
    vis_path = rag_pipeline.visualize_embeddings()
    return "Vector DB created successfully!", vis_path

def ask_question(query, model_type):
    if not rag_pipeline:
        return "Upload files first to build DB.", None
    rag_pipeline.model_source = model_type
    prompt = rag_pipeline.generate_prompt(query)
    response = rag_pipeline.query_llm(prompt)
    return response, None

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Local + OpenAI RAG App - PDF Q&A with Visualization")

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=["ollama", "openai"],
            value="ollama",
            label="Choose LLM Model"
        )

    with gr.Row():
        pdf_upload = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDFs")
        create_btn = gr.Button("Create Vector DB")

    output_status = gr.Textbox(label="Status")
    vis_output = gr.Image(label="Embedding Plot")

    create_btn.click(create_vector_db, inputs=[pdf_upload, model_selector], outputs=[output_status, vis_output])

    with gr.Row():
        query_input = gr.Textbox(label="Ask a question")
        query_btn = gr.Button("Ask")

    query_output = gr.Textbox(label="Answer")

    query_btn.click(ask_question, inputs=[query_input, model_selector], outputs=[query_output])

    gr.Markdown("""
    ### 🔹 Instructions:
    1. Upload PDFs
    2. Choose model: `ollama` (local) or `openai`
    3. Click "Create Vector DB"
    4. Ask any question about the PDFs
    5. View 2D UMAP embedding plot
    """)

demo.launch()