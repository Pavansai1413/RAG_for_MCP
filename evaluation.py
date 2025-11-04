import os
import json
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from langchain_google_genai import ChatGoogleGenerativeAI


# ----------- CONFIG -----------
top_k = 3
output_dir = "evaluation_results"
ground_truth_path = "groundtruth_data.json"


# ----------- LOAD GROUND TRUTH -----------
def load_ground_truth(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load ground truth data: {e}")

    testset = []
    for item in data:
        try:
            question = item["inputs"]["question"]
            answer = item["outputs"]["answer"]
            testset.append({"question": question, "ground_truths": answer})
        except KeyError as e:
            print(f"Skipping item due to missing key: {e}")

    if not testset:
        raise ValueError("No valid test items found in the ground truth data.")
    return testset


# ----------- EVALUATION SYSTEM -----------
def evaluation_system() -> VectorStoreManager:
    doc_processor = DocumentProcessor(chunk_size=600, chunk_overlap=80)
    split_docs = doc_processor.load_and_split_documents()
    print(f"Loaded and split {len(split_docs)} documents for evaluation.")

    vman = VectorStoreManager(index_name="evaluation-index")
    vman.setup_index()
    vman.create_vector_store(split_docs)
    return vman


# ----------- COLLECT RESULTS -----------
def collect_results(vman: VectorStoreManager, testset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_rows = []
    for idx, item in enumerate(testset):
        question = item["question"]

        # Retrieve documents
        retrieved_docs = vman.search(question, top_k=top_k)

        contexts = []
        for doc in retrieved_docs:
            if hasattr(doc, "page_content"):
                contexts.append(doc.page_content)
            elif isinstance(doc, (tuple, list)) and len(doc) > 0:
                contexts.append(str(doc[0]))
            else:
                contexts.append(str(doc))

        # Generate model answer
        answer = vman.generate_answer(question, top_k=top_k)

        all_rows.append({
            "question": question,
            "contexts": contexts,
            "response": answer,                     
            "ground_truths": item["ground_truths"],
            "reference": item["ground_truths"],
        })
    return all_rows


# ----------- SAFE GOOGLE LLM WRAPPER -----------
class SafeChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    def _generate_content(self, *args, **kwargs):
        if "max_retries" in kwargs:
            kwargs.pop("max_retries")
        return super()._generate_content(*args, **kwargs)


# ----------- EVALUATION -----------
def evaluate_results(rows: List[Dict[str, Any]]):
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    llm = SafeChatGoogleGenerativeAI(
        model="gemma-3-12b-it",
        google_api_key=google_api_key,
        temperature=0.2,
        max_output_tokens=512
    )

    ds = Dataset.from_list(rows)
    metrics = [faithfulness, context_precision, context_recall, answer_relevancy]

    results = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=llm,
        raise_exceptions=False,
    )
    return results


# ----------- SAVE RESULTS -----------
def save_evaluation_results(results, rows, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Handle ragas return type
    if hasattr(results, "to_dict"):
        overall_scores = results.to_dict()
    elif hasattr(results, "model_dump"):
        overall_scores = results.model_dump()
    elif hasattr(results, "results"):
        overall_scores = dict(results.results)
    else:
        overall_scores = {}

    overall_path = os.path.join(out_dir, "overall_score.json")
    with open(overall_path, "w") as f:
        json.dump(overall_scores, f, indent=4)
    print(f"Saved overall scores to {overall_path}")

    try:
        df = results.to_pandas()
        original_df = pd.DataFrame(rows)
        df = pd.concat([original_df, df], axis=1)
        per_row_path = os.path.join(out_dir, "per_row_scores.csv")
        df.to_csv(per_row_path, index=False)
        print(f"Saved per-row scores to {per_row_path}")
    except Exception as e:
        print(f"Warning: could not save per-row metrics: {e}")


# ----------- CLEAN ANSWER EXTRACTION -----------
def extract_answer(text: str) -> str:
    """Extracts only the model's actual answer portion."""
    if not isinstance(text, str):
        return ""
    if "###ANSWER_START###" in text:
        text = text.split("###ANSWER_START###")[-1]
    elif "Answer:" in text:
        text = text.split("Answer:")[-1]
    # Clean whitespace and pick first non-empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


# ----------- MAIN -----------
if __name__ == "__main__":
    print("Starting RAG evaluation using Gemma-3-12B-Instruct via Google API...")

    testset = load_ground_truth(ground_truth_path)
    vman = evaluation_system()
    rows = collect_results(vman, testset)
    for row in rows:
        row["response"] = extract_answer(row.get("response", ""))
        if not row["response"].strip():
            row["response"] = "No answer generated."

    results = evaluate_results(rows)
    save_evaluation_results(results, rows, output_dir)

    print("Evaluation complete!")
    print("Overall scores:", results)
