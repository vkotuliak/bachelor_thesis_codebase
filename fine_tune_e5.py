from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer.losses import (
    MultipleNegativesRankingLoss,
)
from torch.utils.data import DataLoader
import json

PATH = "data/full_data/documents_w_queries.jsonl"
model_id = "intfloat/e5-large-v2"
model = SentenceTransformer(model_id)


def load_corpus(path: str):
    examples = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            q = f"query: {item['query']}"
            parts = [
                item.get("name", ""),
                item.get("aliases", ""),
                item.get("item description", ""),
                item.get("typical applications", ""),
                item.get("tags", ""),
            ]
            a = "passage: " + "[SEP]".join(p for p in parts if p)
            examples.append(InputExample(texts=[q, a]))
    return examples


training_examples = load_corpus(PATH)
train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=16)

train_loss = MultipleNegativesRankingLoss(model=model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True,
    output_path="./fine-tuned-e5-model",
)

print("Fine-tuning complete. Model saved to ./fine-tuned-e5-model")
