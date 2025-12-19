import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

MODEL = "facebook/contriever"
DEVICE = "cuda" if torch.cuda.is_available() else "mps"

BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(DEVICE)

# Freeze doc encoder → faster
for p in model.parameters():
    p.requires_grad = False
for p in model.encoder.layer[-2:].parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)

with open("adore_train.json") as f:
    data = json.load(f)

def encode(texts):
    toks = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt"
    ).to(DEVICE)
    return model(**toks).last_hidden_state[:, 0]

loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

for _ in range(EPOCHS):
    for batch in loader:
        q = encode(batch["query"])
        pos = encode([p["text"] for p in batch["positive"]])
        neg = encode([n["text"] for b in batch["negatives"] for n in b])

        logits = torch.matmul(q, torch.cat([pos, neg]).T)
        labels = torch.arange(len(q)).to(DEVICE)

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

torch.save(model.state_dict(), "adore.pt")
print("Saved adore.pt")