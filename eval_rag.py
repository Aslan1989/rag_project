test_case = {
    "question": "What three things does NovaSoft's technical support team help restaurant staff with?",
    "expected_chunk_ids": {"sample.pdf-p3-c1", "sample.pdf-p3-c2"},
}

def retrieval_recall(expected_ids, retrieved_ids):
    found = expected_ids.intersection(retrieved_ids)
    return len(found) / len(expected_ids)

expected = test_case["expected_chunk_ids"]

search_hits = {"sample.pdf-p3-c1"}
final_context = {"sample.pdf-p3-c1", "sample.pdf-p3-c2"}

print(f"Search recall: {retrieval_recall(expected, search_hits):.0%}")
print(f"Final context recall: {retrieval_recall(expected, final_context):.0%}")