from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def retrieve_similar_docs(query, k=4, model_name='all-MiniLM-L6-v2', store_path='faiss_store'):
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.load_local(store_path, embedding, allow_dangerous_deserialization=True)
    return vectorstore.similarity_search(query, k=k)

def retrieve_with_mmr(query, k=4, fetch_k=8, model_name='all-MiniLM-L6-v2', store_path='faiss_store'):
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.load_local(store_path, embedding, allow_dangerous_deserialization=True)
    return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)