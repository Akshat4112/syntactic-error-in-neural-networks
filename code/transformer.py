import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification

os.environ["WANDB_DISABLED"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    str(PROJECT_ROOT / "models" / "training_model_bert_full_data")
)
print("Model loaded successfully")

# 0 = singular (VBZ), 1 = plural (VBP)
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=device)
print(classifier("My all friends "))
