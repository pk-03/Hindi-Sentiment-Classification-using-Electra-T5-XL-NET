# Hindi Sentiment Classification using ELECTRA, T5, and XLNet

This repository contains code and configuration for fine-tuning multiple transformer models — **ELECTRA**, **T5**, and **XLNet** — for **sentiment classification** of Hindi text. The models are evaluated and compared based on standard classification metrics.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Models Used](#-models-used)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference](#-inference)
- [Results](#-results)
- [License](#-license)

---

## 🧠 Overview

This project fine-tunes and evaluates three state-of-the-art transformer architectures for **Hindi sentiment analysis**:

- ELECTRA (`google/electra-base-discriminator`)
- T5 (`google/mt5-base` or any multilingual T5 variant)
- XLNet (`xlnet-base-cased`)

Key features:
- Text classification on Hindi data
- Preprocessing and tokenization using Hugging Face Transformers
- Fine-tuning with `Trainer`
- Evaluation on standard metrics
- Batch inference support

---

## 🧪 Models Used

| Model      | Hugging Face ID                      | Notes                               |
|------------|---------------------------------------|-------------------------------------|
| ELECTRA    | `google/electra-base-discriminator`   | Discriminator-only model            |
| T5         | `google/mt5-base` or `t5-base`        | Encoder-decoder; needs special handling for classification |
| XLNet      | `xlnet-base-cased`                    | Permutation-based autoregressive LM |


## 🚀 Training
Each model can be trained with a corresponding script or configuration block.

1. Load your dataset:
```bash
import pandas as pd
df = pd.read_csv("your_dataset.csv")
```

2. Choose a model and run the training loop using Trainer. Each script includes:

- Tokenization
- Dataset wrapping
- Model initialization
- Metric computation
- Early stopping and checkpointing

## 📈 Evaluation
Metrics computed after training:
- Accuracy
- F1 Score (weighted)
- Precision (weighted)
- Recall (weighted)

Example:
```bash
trainer.evaluate(test_dataset)
```

## 🔍 Inference
To predict the sentiment of a custom Hindi sentence:
```bash
def classify_text(text, tokenizer, model, label_mapping, device):
    tokens = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**tokens)
    prediction = torch.argmax(outputs.logits, dim=1)
    return list(label_mapping.keys())[prediction.item()]
```

Example:
```bash
sample_text = "यह फिल्म मजेदार थी!"
print(classify_text(sample_text, tokenizer, model, label_mapping, device))
```


## 📊 Results (Sample)

Electra    

### Results comparision of the given model on Product reveiws dataset
|Model|Accuracy|F1 Score|Precision|Recall|
|ELECTRA|	0.4231|	0.2515|	0.179	|0.4231|
|T5|	0.694|	0.705 |	0.743	|0.694|
|XLNet|	0.4796|	0.3877|	0.5158	|0.4796|

### Results comparision of the given model on Movies reveiws dataset
|Model|Accuracy|F1 Score|Precision|Recall|
|-----|--------|--------|---------|-------|
|ELECTRA|	0.49|	0.45|	0.50	|0.4993|
|T5|	0.6706|	0.668|	0.6747	|0.67|
|XLNet|	0.41|	0.24|	0.18	|0.41|

Note: Results will vary based on dataset and hyperparameters.

## 💾 Saving and Loading
To save a trained model:
```bash
model.save_pretrained("model-dir")
tokenizer.save_pretrained("model-dir")
```

> To reload later:
```bash
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("model-dir")
tokenizer = AutoTokenizer.from_pretrained("model-dir")
```

## 🧪 Notes on T5
T5 is originally a sequence-to-sequence model. For classification:

Input is: "classify: <text>"

Target label is treated as a string ("Positive", "Negative"...)

You may need to customize the model and loss function for classification tasks if using raw T5.

## 📄 License
This project is licensed under the MIT License.

## 🙋‍♀️ Contact
For questions, suggestions, or contributions, please open an issue or reach out to me directly.
