import os
import httpx
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL")
GROQ_MODEL = os.getenv("GROQ_MODEL")

def standard_prompt(context, query):
    return f"""You are an expert Artificial Neural Networks assistant. Use the following context to answer the question clearly and concisely.

Context:
{context}

Question:
{query}

Answer:"""

def alternative_prompt(context, query):
    return f"""Answer the question below using the provided documents.

Docs:
{context}

Q: {query}
A:"""

###BONUS
def rewrite_query(query):
    rewrite_prompt = f"""You are an Artifical Neural Networks assistant helping to improve information retrieval. 

Rewrite the user's question to make it clearer and more focused, without changing its meaning.

Original question: "{query}"

Rewritten question:"""

    return generate_with_groq(rewrite_prompt)


def generate_with_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful and knowledgable assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    try:
        response = httpx.post(f"{GROQ_BASE_URL}/chat/completions", headers=headers, json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("Error:", e)
        return "Generation failed."

def run_rag(query, k=4, model_name="all-MiniLM-L6-v2", store_path="faiss_store", use_alternative_prompt=False, rewrite=False):
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.load_local(store_path, embedding, allow_dangerous_deserialization=True)

    if rewrite:
        query = rewrite_query(query)

    retrieved_docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = alternative_prompt(context, query) if use_alternative_prompt else standard_prompt(context, query)
    response = generate_with_groq(prompt)

    print(f"\n Query: {query}\n")
    print(f" Answer:\n{response}")
    return response