# RagSystems-Lab6-
Neural Networks RAG System – CSAI 422 Lab 6
This project builds a Retrieval-Augmented Generation (RAG) system focused on the domain of Neural Networks. It retrieves relevant document chunks using semantic search and enhances LLM-generated answers using that context.

Setup Instructions
1. Create and activate a virtual environment:
python -m venv venv
Activate with .\venv\Scripts\activate (on Windows)

2. Install dependencies:
pip install -r requirements.txt

3. Run the system:
python testing.py

System Architecture
- Document loading using LangChain community loaders
- Text splitting and embedding using Sentence Transformers (e.g., MiniLM)
- FAISS vector store for local similarity search
- Two retrieval modes: basic similarity and Max Marginal Relevance (MMR)
- Retrieved chunks are passed into a prompt for LLM generation

Results from Retrieval Strategies
Two strategies were compared using precision, recall, and F1 score:
- Similarity Search: Precision 0.85, Recall 0.78, F1 Score 0.81
- MMR (fetch_k=8): Precision 0.80, Recall 0.83, F1 Score 0.81
MMR produced more diverse results, while similarity search was more focused.

Strengths and Weaknesses
Strengths:
- Local, private vector storage
- Fast, accurate retrieval
- Modular design for easy experimentation

Weaknesses:
- FAISS can consume memory with large corpora
- LLM output depends on input document quality

Challenges and Solutions
- Faced deprecation warnings from LangChain v0.2 → fixed by updating imports to langchain_community
- Missing modules like langchain or rag → resolved by installing required packages
- Import error in custom script → fixed function definitions and file structure

Project Structure
Documents/: Source documents on neural networks
documentProcessing.py: Loading and chunking documents
retrieval.py: Similarity and MMR search
evaluation.py: Metrics and performance analysis
comparingConfiguration: Comparing configurations
RAGSystem.py: Initializing ChatBot
testing.py: Pipeline runner
requirements.txt: List of dependencies

Author: Mira Emad
Course: CSAI 422 – Lab 6
Topic: Neural Networks RAG System
Date: April 2025
