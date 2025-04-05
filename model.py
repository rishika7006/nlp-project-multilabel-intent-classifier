# model.py

from transformers import XLNetTokenizer, XLNetForSequenceClassification
import torch

def load_model(model_path="trained_model"):
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    model = XLNetForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits).squeeze().tolist()
    return probabilities