"""
Retry rate-limited queries from prediction JSON files.

Scans prediction files, identifies 429 rate limit errors,
and retries those queries with proper backoff.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from groq import Groq
from tqdm import tqdm

# Import functions from run_negative_experiments
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# API keys
API_KEYS = [os.getenv(f"GROQ_API_KEY{i if i > 1 else ''}") 
            for i in range(1, 11)]
API_KEYS = [k for k in API_KEYS if k and k.strip()]

if not API_KEYS:
    raise ValueError("No API keys found in .env file")

print(f"Loaded {len(API_KEYS)} API key(s)")

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
    if not os.path.exists(filepath):
        json_files_path = os.path.join('json_files', filepath)
        if os.path.exists(json_files_path):
            filepath = json_files_path
        else:
            raise FileNotFoundError(f"Retrieval file not found: {filepath}")
    
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


def call_llm_with_backoff(client, system_prompt, user_prompt, max_retries=5, initial_delay=1):
    """Call LLM with exponential backoff for rate limits"""
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
            error_str = str(e)
            # Check if it's a rate limit error
            if '429' in error_str or 'rate_limit' in error_str.lower() or 'rate limit' in error_str.lower():
                if attempt < max_retries - 1:
                    # Extract wait time from error if available
                    wait_time = initial_delay * (2 ** attempt)
                    # Try to extract suggested wait time from error message
                    wait_match = re.search(r'try again in ([\d.]+)ms', error_str, re.IGNORECASE)
                    if wait_match:
                        wait_time = float(wait_match.group(1)) / 1000.0  # Convert ms to seconds
                        wait_time = max(wait_time, 0.1)  # Minimum 100ms
                    else:
                        wait_time = min(wait_time, 60)  # Cap at 60 seconds
                    
                    print(f"  Rate limited, waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"ERROR: Rate limit exceeded after {max_retries} retries: {error_str}"
            else:
                # Non-rate-limit error, return immediately
                return f"ERROR: {error_str}"
    
    return f"ERROR: Max retries exceeded"


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
        
        # Call LLM with backoff
        user_prompt = f"[Context]:\n{context_str}\n\n[Question]: {question_text}"
        answer = call_llm_with_backoff(client, SYSTEM_PROMPT, user_prompt)
        
        return q_id, answer
        
    except Exception as e:
        return q_id, f"ERROR: {str(e)}"


def parse_filename(filename):
    """Parse experiment parameters from filename"""
    # Format: predictions_random_k1_ratio_1_5.json
    # or: predictions_hard_k3_ratio_3_6.json
    match = re.match(r'predictions_(random|hard)_k(\d+)_ratio_(\d+)_(\d+)\.json', filename)
    if match:
        negative_type = match.group(1)
        k = int(match.group(2))
        R = int(match.group(3))
        N = int(match.group(4))
        ratio = (R, N)
        return k, ratio, negative_type
    return None, None, None


def find_rate_limited_queries(pred_file):
    """Find all rate-limited queries in a prediction file"""
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    rate_limited = []
    for q_id, answer in predictions.items():
        if isinstance(answer, str) and ('ERROR: Error code: 429' in answer or 
                                       'rate_limit_exceeded' in answer or
                                       ('429' in answer and 'rate limit' in answer.lower())):
            rate_limited.append(q_id)
    
    return rate_limited, predictions


def retry_rate_limited(pred_file, questions_map, corpus, oracle_map, 
                       k, ratio, negative_type, retrieval_file, max_workers=None, max_iterations=50):
    """
    Retry rate-limited queries for a specific prediction file.
    Keeps retrying until all rate-limited errors are fixed or max_iterations reached.
    """
    
    # Adjust workers based on number of API keys
    if max_workers is None:
        if len(API_KEYS) == 1:
            max_workers = 5  # Lower for single key to avoid rate limits
        else:
            max_workers = 10  # Default for multiple keys
    
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(pred_file)}")
    print(f"k={k}, ratio={ratio[0]}:{ratio[1]}, type={negative_type}")
    print(f"{'='*80}")
    
    # Load retrieval results once
    retrieval_results = load_retrieval(retrieval_file)
    
    total_updated = 0
    iteration = 0
    
    # Keep retrying until all rate-limited queries are fixed
    while iteration < max_iterations:
        iteration += 1
        
        # Find rate-limited queries (reload file each iteration to get latest state)
        rate_limited, predictions = find_rate_limited_queries(pred_file)
        
        if not rate_limited:
            print(f"\n✓ All rate-limited queries fixed! File is complete!")
            return total_updated
        
        print(f"\n--- Iteration {iteration} ---")
        print(f"Found {len(rate_limited)} rate-limited queries to retry")
        
        # Prepare tasks
        tasks = []
        for idx, q_id in enumerate(rate_limited):
            if q_id not in retrieval_results or q_id not in questions_map:
                print(f"  Warning: Skipping {q_id} (not in retrieval or questions)")
                continue
            
            question_text = questions_map[q_id]['question']
            retrieved_docs = list(retrieval_results[q_id].keys())
            oracle_docs = oracle_map.get(q_id, set())
            
            # Round-robin API key assignment
            api_key = API_KEYS[idx % len(API_KEYS)]
            
            tasks.append((q_id, question_text, retrieved_docs, oracle_docs, corpus,
                         k, ratio, negative_type, retrieval_results, oracle_map, api_key))
        
        if not tasks:
            print("No valid tasks to process")
            break
        
        print(f"Retrying {len(tasks)} queries with {max_workers} workers...")
        
        # Process with lower concurrency to avoid rate limits
        updated_count = 0
        still_rate_limited = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_question, *task): task[0] for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                              desc=f"Retrying (iter {iteration})"):
                q_id, answer = future.result()
                
                # Only update if we got a non-error answer
                if not answer.startswith("ERROR"):
                    predictions[q_id] = answer
                    updated_count += 1
                elif '429' in answer or 'rate_limit' in answer.lower():
                    # Still rate-limited, keep the error for next iteration
                    still_rate_limited += 1
                else:
                    # Different error (not rate limit), update to avoid infinite retry
                    predictions[q_id] = answer
                    updated_count += 1
        
        # Save updated predictions after each iteration
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        total_updated += updated_count
        print(f"\nIteration {iteration} results:")
        print(f"  Updated: {updated_count} queries")
        print(f"  Still rate-limited: {still_rate_limited} queries")
        print(f"  Total updated so far: {total_updated}")
        
        # If no progress was made and still have rate-limited queries, wait before next iteration
        if still_rate_limited > 0 and updated_count == 0:
            wait_time = min(60, 10 * iteration)  # Wait longer on later iterations
            print(f"\nNo progress made. Waiting {wait_time}s before next iteration...")
            time.sleep(wait_time)
        elif still_rate_limited == 0:
            # All fixed!
            break
    
    # Final check
    rate_limited, predictions = find_rate_limited_queries(pred_file)
    remaining = len(rate_limited)
    
    if remaining > 0:
        print(f"\n⚠ Warning: {remaining} rate-limited queries remain after {iteration} iterations")
        print(f"  This may be due to persistent rate limiting. You may need to:")
        print(f"  - Wait longer between retries")
        print(f"  - Reduce max_workers (currently {max_workers})")
        print(f"  - Check API rate limits")
    else:
        print(f"\n✓ Successfully fixed all rate-limited queries in {iteration} iterations!")
    
    return total_updated


def main():
    """Main retry script"""
    
    print("="*80)
    print("RETRY RATE-LIMITED QUERIES")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    questions_map, corpus = load_data()
    oracle_map = get_oracle_docs(questions_map, corpus)
    
    # Limit to test set
    test_ids = list(questions_map.keys())[:1200]
    questions_map = {q_id: questions_map[q_id] for q_id in test_ids}
    oracle_map = {q_id: oracle_map[q_id] for q_id in test_ids if q_id in oracle_map}
    
    print(f"Test set: {len(questions_map)} questions")
    
    # Find all prediction files
    results_dir = Path('results_negative_experiments')
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return
    
    pred_files = list(results_dir.glob('predictions_*.json'))
    
    if not pred_files:
        print(f"No prediction files found in {results_dir}")
        return
    
    print(f"\nFound {len(pred_files)} prediction files")
    
    # Process each file
    total_updated = 0
    files_processed = 0
    
    for pred_file in sorted(pred_files):
        # Parse parameters from filename
        k, ratio, negative_type = parse_filename(pred_file.name)
        
        if k is None:
            print(f"\nSkipping {pred_file.name} (could not parse parameters)")
            continue
        
        # Determine retrieval file
        retrieval_file = f'retrieval_k{k}.json'
        
        # Retry rate-limited queries (will keep retrying until all fixed)
        updated = retry_rate_limited(
            str(pred_file), questions_map, corpus, oracle_map,
            k, ratio, negative_type, retrieval_file,
            max_workers=None,  # Will auto-adjust based on API keys
            max_iterations=50  # Maximum iterations per file (prevents infinite loops)
        )
        
        total_updated += updated
        files_processed += 1
        
        # Small delay between files to avoid overwhelming the API
        if files_processed < len(pred_files):
            time.sleep(2)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Files processed: {files_processed}")
    print(f"Total queries updated: {total_updated}")
    
    # Check if any files still have rate-limited queries
    remaining_files = []
    for pred_file in pred_files:
        rate_limited, _ = find_rate_limited_queries(str(pred_file))
        if rate_limited:
            remaining_files.append((pred_file.name, len(rate_limited)))
    
    if remaining_files:
        print(f"\n⚠ Warning: {len(remaining_files)} file(s) still have rate-limited queries:")
        for filename, count in remaining_files:
            print(f"  - {filename}: {count} queries")
        print(f"\nThe script will retry these automatically on the next run.")
    else:
        print(f"\n✓ All files are error-free! No rate-limited queries remaining.")


if __name__ == "__main__":
    main()
