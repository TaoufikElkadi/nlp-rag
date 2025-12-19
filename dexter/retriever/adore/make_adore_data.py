import json
import random

from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset

# =====================
# Config
# =====================
DATASET = "wikimultihopqa"
CORPUS = "wikimultihopqa-corpus"
CONFIG = "dexter/config/config.ini"

MAX_SAMPLES = 500
MAX_NEGS = 3

# =====================
# Load dataset
# =====================
loader = RetrieverDataset(
    DATASET,
    CORPUS,
    CONFIG,
    split=Split.TRAIN,
    tokenizer=None
)

queries, qrels, corpus = loader.qrels()

# =====================
# Load retrieval output
# retrieval_k5.json format:
# qid -> {doc_id: score}
# =====================
with open("retrieval_k5.json") as f:
    contriever = json.load(f)

out = []

# =====================
# Build ADORE samples
# =====================
for q in queries:
    qid = str(q.id())

    if qid not in qrels or qid not in contriever:
        continue

    gold = list(qrels[qid].keys())
    if not gold:
        continue

    # Positive
    pos_id = random.choice(gold)
    if pos_id not in corpus:
        continue

    pos_text = corpus[pos_id]["title"] + " " + corpus[pos_id]["text"]

    # Negatives (hard negatives from retriever)
    negs = []

    ranked_docs = sorted(
        contriever[qid].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for doc_id, _ in ranked_docs:
        if doc_id not in gold and doc_id in corpus:
            negs.append({
                "doc_id": doc_id,
                "text": corpus[doc_id]["title"] + " " + corpus[doc_id]["text"]
            })
        if len(negs) == MAX_NEGS:
            break

    # Require at least one negative
    if len(negs) < 1:
        continue

    out.append({
        "qid": qid,
        "query": q.text,
        "positive": {
            "doc_id": pos_id,
            "text": pos_text
        },
        "negatives": negs
    })

    if len(out) >= MAX_SAMPLES:
        break

# =====================
# Save
# =====================
with open("adore_train.json", "w") as f:
    json.dump(out, f, indent=2)

print("Saved", len(out), "ADORE samples")