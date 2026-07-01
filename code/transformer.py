import os
from pathlib import Path

from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification

os.environ["WANDB_DISABLED"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    str(PROJECT_ROOT / "models" / "training_model_bert_full_data")
)
print("Model loaded successfully")

# 0 = singular (VBZ), 1 = plural (VBP)
classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=0)
print(classifier("My all friends "))
