import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification

def main(project_root=None):
    os.environ.setdefault("WANDB_DISABLED", "true")
    project_root = Path(project_root or Path.cwd()).resolve()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(project_root / "models" / "training_model_bert_full_data")
    )
    print("Model loaded successfully")

    # 0 = singular (VBZ), 1 = plural (VBP)
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=device)
    print(classifier("My all friends "))


if __name__ == "__main__":
    main()
