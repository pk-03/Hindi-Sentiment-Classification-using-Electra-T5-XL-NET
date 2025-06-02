import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Trainer, TrainingArguments, EarlyStoppingCallback


data = pd.read_csv('./../../../augmented_product_reviews.csv')


# Split the data into training and testing
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["Reviews"].tolist(), data["labels"].tolist(), test_size=0.2, random_state=42
)

# Load mT5 tokenizer
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

# Tokenize the text
def tokenize_data(texts, labels):
    inputs = [f"Classify: {text}" for text in texts]
    encodings = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    labels_encodings = tokenizer(
        labels, padding=True, truncation=True, max_length=32, return_tensors="pt"
    )["input_ids"]
    return encodings, labels_encodings

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

# Load pretrained mT5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
model.to(device)

# Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    acc = accuracy_score(decoded_labels, decoded_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        decoded_labels, decoded_preds, average="weighted", zero_division=0
    )
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
    num_train_epochs=50,  # Run for up to 50 epochs
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
    seed=42,
    push_to_hub=False,
)

# Initialize the Trainer with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],  # Early stopping
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results:", results)

# Save the model
trainer.save_model("./mt5-hindi-classifier")
tokenizer.save_pretrained("./mt5-hindi-classifier")

# Inference: Classify new Hindi text
def classify_text(text):
    model.eval()
    input_text = f"Classify: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=32)
    pred_class = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return pred_class

# Example: Classify a new sentence
new_text = "यह एक शानदार अनुभव था।"
pred_class = classify_text(new_text)
print(f"Predicted Class: {pred_class}")



