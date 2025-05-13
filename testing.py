
# Test Document Processing (Load, Split, Embed)
# ===============================
try:
    from documentProcessing import load_documents, split_documents, embed_and_store

    print("\n Running: documentProcessing.py")
    docs = load_documents('Documents')
    chunks = split_documents(docs)
    embed_and_store(chunks, model_name='all-MiniLM-L6-v2', save_path='faiss_store')
    embed_and_store(chunks, model_name='paraphrase-MiniLM-L12-v2', save_path='faiss_store_l12')
except Exception as e:
    print("documentProcessing.py failed:", e)

"""
# Test Retrieval (with Metadata)

try:
    from retrieval import retrieve_similar_docs, retrieve_with_mmr

    print("\n Running: retrieval.py")

    results = retrieve_similar_docs("What are some hyperparameters to tune for CNN?")
    print(f"Top {len(results)} basic results:")
    for i, doc in enumerate(results, 1):
        print(f"--- Basic Result #{i} ---")
        print("Metadata:", doc.metadata)
        print("Snippet:", doc.page_content[:200])
        print()

    mmr_results = retrieve_with_mmr("What are some hyperparameters to tune for CNN?")
    print(f"Top {len(mmr_results)} MMR results:")
    for i, doc in enumerate(mmr_results, 1):
        print(f"--- MMR Result #{i} ---")
        print("Metadata:", doc.metadata)
        print("Snippet:", doc.page_content[:200])
        print()

except Exception as e:
    print(" retrieval.py failed:", e)



# Test RAG Generation
# ===============================
try:
    from RAGSystem import run_rag

    print("\n Running: RAGSystem.py")
    run_rag("What are some hyperparameters to tune for CNN?")
    run_rag("What are some strategies for trainging neural networks?", use_alternative_prompt=True)
    run_rag("What are some real life applications for Neural Networks?", model_name="paraphrase-MiniLM-L12-v2")
    run_rag("What are RNNs?", rewrite=True)
except Exception as e:
    print(" rag.py failed:", e)


#  Test Evaluation (Real RAG Output + Real Retrieval IDs)
# ===============================
try:
    from retrieval import retrieve_similar_docs
    from RAGSystem import run_rag
    from evaluation import evaluate_answer_quality, evaluate_retrieval

    print("\n Running: evaluation.py")

    # Define a real test query
    query = ""What are some hyperparameters to tune for CNN?""
    expected_keywords = ["hyper-parameters", "tuning", "learning rate", "batch size", "CNN", "kernel_size", "regularization", "dropout"]

    # Step 1: Run RAG to get generated answer (for answer quality evaluation)
    generated_answer = run_rag(query)
    evaluate_answer_quality(generated_answer, expected_keywords)

    # Step 2: Run retrieval to get actual document metadata
    retrieved_docs = retrieve_similar_docs(query, k=4)
    retrieved_ids = [doc.metadata.get("source", f"doc{i}") for i, doc in enumerate(retrieved_docs)]

    
    relevant_ids = retrieved_ids[:3]  #actual relevant files

    # Step 4: Evaluate retrieval
    evaluate_retrieval(relevant_ids, retrieved_ids)

except Exception as e:
    print(" evaluation.py failed:", e)"""


#  Test Configuration Comparison
# ===============================
try:
    import comparingConfiguration
except Exception as e:
    print("comparingConfiguration.py failed:", e)
