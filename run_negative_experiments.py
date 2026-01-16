"""
Negative sampling experiments for RAG on 2WikiMultiHopQA.

Tests both random and hard negative contexts with various ratios.
Uses parallel processing to speed up experiments.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import random
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq
from tqdm import tqdm

# API keys
API_KEYS = [os.getenv(f"GROQ_API_KEY{i if i > 1 else ''}") 
            for i in range(1, 11)]
API_KEYS = [k for k in API_KEYS if k and k.strip()]

if not API_KEYS:
    raise ValueError("No API keys found in .env file")

print(f"Loaded {len(API_KEYS)} API key(s)")

# No rate limiting
DEFAULT_MAX_WORKERS = 20

# Model config
MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0.3
MAX_TOKENS = 256

SYSTEM_PROMPT = """You are a precise answering assistant. You will be provided with a [Question] and a [Context] to help you answer the question. The [Context] consists of documents with a [Title] and [Text].
1. Answer the question using ONLY the provided context.
2. Do not use full sentences. Provide only the exact answer entity or phrase.
3. Do not add conversational filler like "The answer is" or "Based on the context".
4. If the answer is not contained within the context, respond with "Not Answerable"."""


def load_data():
    """Load questions and corpus"""
    with open('data/dev.json', 'r') as f:
        questions = json.load(f)
    questions_map = {q['_id']: q for q in questions}
    
    with open('data/wiki_musique_corpus.json', 'r') as f:
        corpus = json.load(f)
    
    return questions_map, corpus


def load_retrieval(filepath):
    """Load retrieval results"""
    # Try json_files directory first, then root
    if not os.path.exists(filepath):
        json_files_path = os.path.join('json_files', filepath)
        if os.path.exists(json_files_path):
            filepath = json_files_path
        else:
            raise FileNotFoundError(f"Retrieval file not found: {filepath} or {json_files_path}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def get_oracle_docs(questions_map, corpus):
    """Extract oracle document IDs by matching titles"""
    title_to_doc = {}
    for doc_id, doc in corpus.items():
        title = doc.get('title', '').strip()
        if title:
            if title not in title_to_doc:
                title_to_doc[title] = []
            title_to_doc[title].append(doc_id)
    
    oracle_map = {}
    for q_id, q_data in questions_map.items():
        oracle_docs = set()
        if 'context' in q_data:
            for ctx in q_data['context']:
                title = ctx[0].strip() if isinstance(ctx, list) else str(ctx).strip()
                if title in title_to_doc:
                    oracle_docs.update(title_to_doc[title])
                else:
                    # Case-insensitive match
                    for t, ids in title_to_doc.items():
                        if t.lower() == title.lower():
                            oracle_docs.update(ids)
                            break
        oracle_map[q_id] = oracle_docs
    
    return oracle_map


def sample_random_negatives(corpus, excluded, n):
    """Sample n random documents excluding specified ones"""
    available = list(set(corpus.keys()) - excluded)
    if len(available) < n:
        return available
    return random.sample(available, n)


def sample_hard_negatives(retrieval_results, oracle_map, q_id, k, n, max_rank=50):
    """Sample hard negatives: similar but not in oracle"""
    if q_id not in retrieval_results:
        return []
    
    all_retrieved = list(retrieval_results[q_id].keys())
    oracle_docs = oracle_map.get(q_id, set())
    top_k = set(all_retrieved[:k])
    excluded = oracle_docs | top_k
    
    hard_negatives = []
    for doc_id in all_retrieved[k:max_rank]:
        if doc_id not in excluded:
            hard_negatives.append(doc_id)
        if len(hard_negatives) >= n:
            break
    
    return hard_negatives[:n]


def combine_contexts(relevant_docs, negative_docs, ratio):
    """Combine relevant and negative docs according to ratio"""
    R, N = ratio
    selected_relevant = relevant_docs[:R] if R <= len(relevant_docs) else relevant_docs
    selected_negatives = negative_docs[:N] if N <= len(negative_docs) else negative_docs
    combined = selected_relevant + selected_negatives
    random.shuffle(combined)
    return combined


def call_llm(client, system_prompt, user_prompt, max_retries=3):
    """Call LLM with retry logic"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {str(e)}"
    return "ERROR: Max retries exceeded"


def process_question(q_id, question_text, retrieved_docs, oracle_docs, corpus,
                     k, ratio, negative_type, retrieval_results, oracle_map, api_key):
    """Process a single question"""
    try:
        client = Groq(api_key=api_key)
        
        # Get relevant docs
        relevant = retrieved_docs[:k] if len(retrieved_docs) >= k else retrieved_docs
        
        # Sample negatives
        R, N = ratio
        excluded = set(relevant) | oracle_docs
        
        if negative_type == 'random':
            negatives = sample_random_negatives(corpus, excluded, N)
        elif negative_type == 'hard':
            negatives = sample_hard_negatives(retrieval_results, oracle_map, q_id, k, N)
        else:
            negatives = []
        
        # Combine contexts
        combined = combine_contexts(relevant, negatives, ratio)
        
        # Build context string
        context_str = ""
        for doc_id in combined:
            if doc_id in corpus:
                doc = corpus[doc_id]
                context_str += f"[Title]: {doc.get('title', '')}\n[Text]: {doc.get('text', '')}\n\n"
        
        if not context_str.strip():
            return q_id, "ERROR: Empty context"
        
        # Call LLM
        user_prompt = f"[Context]:\n{context_str}\n\n[Question]: {question_text}"
        answer = call_llm(client, SYSTEM_PROMPT, user_prompt)
        
        return q_id, answer
        
    except Exception as e:
        return q_id, f"ERROR: {str(e)}"


def run_experiment(retrieval_file, questions_map, corpus, oracle_map, 
                   k, ratio, negative_type, output_file, max_workers=None):
    """Run a single experiment"""
    
    if max_workers is None:
        max_workers = DEFAULT_MAX_WORKERS
    
    retrieval_results = load_retrieval(retrieval_file)
    question_ids = list(questions_map.keys())[:1200]
    
    # Prepare tasks for parallel processing
    tasks = []
    for idx, q_id in enumerate(question_ids):
        if q_id not in retrieval_results or q_id not in questions_map:
            continue
        
        question_text = questions_map[q_id]['question']
        retrieved_docs = list(retrieval_results[q_id].keys())
        oracle_docs = oracle_map.get(q_id, set())
        
        # Round-robin API key assignment (works with single key too)
        api_key = API_KEYS[idx % len(API_KEYS)]
        
        tasks.append((q_id, question_text, retrieved_docs, oracle_docs, corpus,
                     k, ratio, negative_type, retrieval_results, oracle_map, api_key))
    
    print(f"Processing {len(tasks)} questions with {max_workers} workers")
    
    # Run in parallel
    predictions = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_question, *task): task[0] for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc=f"k={k}, {ratio[0]}:{ratio[1]}, {negative_type}"):
            q_id, answer = future.result()
            predictions[q_id] = answer
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Saved {len(predictions)} predictions to {output_file}")
    return predictions


def main():
    """Main experiment runner"""
    
    print("Loading data...")
    questions_map, corpus = load_data()
    oracle_map = get_oracle_docs(questions_map, corpus)
    
    # Limit to test set
    test_ids = list(questions_map.keys())[:1200]
    questions_map = {q_id: questions_map[q_id] for q_id in test_ids}
    oracle_map = {q_id: oracle_map[q_id] for q_id in test_ids if q_id in oracle_map}
    
    print(f"Test set: {len(questions_map)} questions")
    
    # Experiment configurations
    # Testing wider range of ratios including high ones
    # Based on Cuconasu's pattern: k=1 peaks at 18, k=3 peaks at 16, k=5 peaks at 10
    experiments = [
        # Random negatives - k=1
        (1, (1, 1), 'random', 'retrieval_k1.json'),
        (1, (1, 2), 'random', 'retrieval_k1.json'),
        (1, (1, 5), 'random', 'retrieval_k1.json'),
        (1, (1, 10), 'random', 'retrieval_k1.json'),
        (1, (1, 15), 'random', 'retrieval_k1.json'),
        (1, (1, 16), 'random', 'retrieval_k1.json'),
        (1, (1, 17), 'random', 'retrieval_k1.json'),
        (1, (1, 18), 'random', 'retrieval_k1.json'),
        (1, (1, 19), 'random', 'retrieval_k1.json'),
        (1, (1, 20), 'random', 'retrieval_k1.json'),
        (1, (1, 25), 'random', 'retrieval_k1.json'),

        # Random negatives - k=3
        (3, (3, 1), 'random', 'retrieval_k3.json'),
        (3, (3, 3), 'random', 'retrieval_k3.json'),
        (3, (3, 6), 'random', 'retrieval_k3.json'),
        (3, (3, 10), 'random', 'retrieval_k3.json'),
        (3, (3, 15), 'random', 'retrieval_k3.json'),
        (3, (3, 16), 'random', 'retrieval_k3.json'),
        (3, (3, 17), 'random', 'retrieval_k3.json'),
        (3, (3, 18), 'random', 'retrieval_k3.json'),
        (3, (3, 19), 'random', 'retrieval_k3.json'),
        (3, (3, 20), 'random', 'retrieval_k3.json'),
        (3, (3, 25), 'random', 'retrieval_k3.json'),

        # Random negatives - k=5
        (5, (5, 2), 'random', 'retrieval_k5.json'),
        (5, (5, 5), 'random', 'retrieval_k5.json'),
        (5, (5, 10), 'random', 'retrieval_k5.json'),
        (5, (5, 14), 'random', 'retrieval_k5.json'),
        (5, (5, 15), 'random', 'retrieval_k5.json'),
        (5, (5, 16), 'random', 'retrieval_k5.json'),
        (5, (5, 17), 'random', 'retrieval_k5.json'),
        (5, (5, 18), 'random', 'retrieval_k5.json'),
        (5, (5, 19), 'random', 'retrieval_k5.json'),
        (5, (5, 20), 'random', 'retrieval_k5.json'),
        (5, (5, 25), 'random', 'retrieval_k5.json'),

        # Hard negatives - k=1
        (1, (1, 1), 'hard', 'retrieval_k1.json'),
        (1, (1, 2), 'hard', 'retrieval_k1.json'),
        (1, (1, 5), 'hard', 'retrieval_k1.json'),
        (1, (1, 10), 'hard', 'retrieval_k1.json'),
        (1, (1, 15), 'hard', 'retrieval_k1.json'),
        (1, (1, 16), 'hard', 'retrieval_k1.json'),
        (1, (1, 17), 'hard', 'retrieval_k1.json'),
        (1, (1, 18), 'hard', 'retrieval_k1.json'),
        (1, (1, 19), 'hard', 'retrieval_k1.json'),
        (1, (1, 20), 'hard', 'retrieval_k1.json'),
        (1, (1, 25), 'hard', 'retrieval_k1.json'),

        # Hard negatives - k=3
        (3, (3, 1), 'hard', 'retrieval_k3.json'),
        (3, (3, 3), 'hard', 'retrieval_k3.json'),
        (3, (3, 6), 'hard', 'retrieval_k3.json'),
        (3, (3, 10), 'hard', 'retrieval_k3.json'),
        (3, (3, 15), 'hard', 'retrieval_k3.json'),
        (3, (3, 16), 'hard', 'retrieval_k3.json'),
        (3, (3, 17), 'hard', 'retrieval_k3.json'),
        (3, (3, 18), 'hard', 'retrieval_k3.json'),
        (3, (3, 19), 'hard', 'retrieval_k3.json'),
        (3, (3, 20), 'hard', 'retrieval_k3.json'),
        (3, (3, 25), 'hard', 'retrieval_k3.json'),

        # Hard negatives - k=5
        (5, (5, 2), 'hard', 'retrieval_k5.json'),
        (5, (5, 5), 'hard', 'retrieval_k5.json'),
        (5, (5, 10), 'hard', 'retrieval_k5.json'),
        (5, (5, 14), 'hard', 'retrieval_k5.json'),
        (5, (5, 15), 'hard', 'retrieval_k5.json'),
        (5, (5, 16), 'hard', 'retrieval_k5.json'),
        (5, (5, 17), 'hard', 'retrieval_k5.json'),
        (5, (5, 18), 'hard', 'retrieval_k5.json'),
        (5, (5, 19), 'hard', 'retrieval_k5.json'),
        (5, (5, 20), 'hard', 'retrieval_k5.json'),
        (5, (5, 25), 'hard', 'retrieval_k5.json'),
    ]
    
    # Create results directory
    results_dir = 'results_negative_experiments'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nRunning {len(experiments)} experiments...")
    print(f"Results will be saved to {results_dir}/")
    
    for k, ratio, neg_type, retrieval_file in experiments:
        R, N = ratio
        output_file = os.path.join(results_dir, f"predictions_{neg_type}_k{k}_ratio_{R}_{N}.json")
        
        # Skip if already done
        if os.path.exists(output_file):
            print(f"Skipping {output_file} (already exists)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Experiment: k={k}, ratio={R}:{N}, type={neg_type}")
        print(f"{'='*60}")
        
        run_experiment(
            retrieval_file, questions_map, corpus, oracle_map,
            k, ratio, neg_type, output_file
        )
        
        # Small delay between experiments
        time.sleep(2)
    
    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
