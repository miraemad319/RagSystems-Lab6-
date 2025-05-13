def evaluate_retrieval(relevant_ids, retrieved_ids):
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)
    true_positives = len(relevant_set & retrieved_set)
    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    recall = true_positives / len(relevant_set) if relevant_set else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    print(f"\n📊 Retrieval Evaluation")
    print(f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}")

def evaluate_answer_quality(generated_answer, expected_keywords):
    matched = sum(1 for word in expected_keywords if word.lower() in generated_answer.lower())
    score = matched / len(expected_keywords) if expected_keywords else 0
    print(f"\n📝 Answer Quality Evaluation")
    print(f"Keyword Match Score: {score:.2f}")
