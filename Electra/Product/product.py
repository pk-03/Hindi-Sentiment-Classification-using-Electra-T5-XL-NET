from transformers import ElectraTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import pandas as pd
import numpy as np

data = pd.read_csv('./../../DATASETS/augmented_product_reviews.csv')
df = data.copy()

label_mapping = {"positive": 2, "neutral": 1, "negative": 0}  # Example mapping
data["labels"] = data["labels"].map(label_mapping)  # Map string labels to integers

# Split the data into training and testing
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["Reviews"].tolist(), data["labels"].tolist(), test_size=0.2, random_state=42
)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Electra tokenizer
tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")

# Tokenize the text
def tokenize_data(texts, labels):
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    return encoding, torch.tensor(labels)

train_encodings, train_labels = tokenize_data(train_texts, train_labels)
test_encodings, test_labels = tokenize_data(test_texts, test_labels)

# Create PyTorch datasets
class HindiDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }

train_dataset = HindiDataset(train_encodings, train_labels)
test_dataset = HindiDataset(test_encodings, test_labels)

# Load pretrained Electra model for classification
model = ElectraForSequenceClassification.from_pretrained(
    "google/electra-small-discriminator", num_labels=3
)
model.to(device)

# Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
    seed=42,
    push_to_hub=False,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=7)],
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results:", results)

# Save the model
trainer.save_model("./electra-hindi-classifier")
tokenizer.save_pretrained("./electra-hindi-classifier")

# Inference: Classify new Hindi text
def classify_text(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_class = torch.argmax(probs, dim=1).item()
    return pred_class, probs.tolist()

# Example: Classify a new sentence
new_text = "यह एक शानदार अनुभव था।"
pred_class, probs = classify_text(new_text)
print(f"Predicted Class: {pred_class}, Probabilities: {probs}")





