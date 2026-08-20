# 🏦 Banking Customer Intent Classification

NLP classification of banking customer queries using a **TF-IDF + MLP baseline** and **RoBERTa fine-tuned with LoRA**, with PII protection integrated into the prediction workflow.

---

## 📌 Overview

This project compares two text classification approaches for predicting customer intent from banking queries:

1. **TF-IDF + Multi-Layer Perceptron (MLP)** as a baseline
2. **RoBERTa + LoRA** for parameter-efficient transformer fine-tuning

The models are evaluated on the **Banking77** dataset, which contains short natural-language banking queries belonging to 77 different intent classes.

The project also implements a PII protection workflow that processes sensitive information before prediction. Credit-card and other sensitive financial information can be redacted, while email addresses and phone numbers can be hashed.

### Project Goals

- Explore and visualize the Banking77 dataset
- Implement a baseline MLP classifier
- Fine-tune a pretrained RoBERTa model using LoRA
- Compare model performance using classification metrics
- Analyze performance at the individual-intent level
- Implement PII protection for customer queries

---

## 📊 Dataset

The project uses the **Banking77** dataset.

### Dataset characteristics

- **77** unique intent classes
- **13,083** total queries
- **10,003** training examples
- **3,080** test examples

The queries represent banking-related customer requests such as card issues, payments, transfers, cash withdrawals, and identity verification.

The dataset was explored using visualizations including:

- Intent-class distributions
- Query-length distributions
- Word clouds for frequently occurring intents

Dataset source:

**Banking77 — PolyAI**

https://huggingface.co/datasets/PolyAI/banking77

---

## 🧠 Modeling Approach

### 1. TF-IDF + MLP Baseline

The baseline converts customer queries into TF-IDF feature vectors before passing them to a PyTorch MLP classifier.

TF-IDF configuration:

- Lowercase text
- Unigrams and bigrams
- `min_df=2`
- `max_df=0.95`
- Maximum 50,000 features
- Unicode accent stripping

The TF-IDF vectorizer is fitted on the training data and then used to transform the test data.

The resulting representation contains **10,292 features**.

The MLP performs multi-class classification across all **77 intents**.

---

### 2. RoBERTa + LoRA

The second approach uses the pretrained **`roberta-base`** model for sequence classification.

Instead of fine-tuning the entire transformer, the project uses **LoRA (Low-Rank Adaptation)** through the PEFT library.

LoRA is applied to the:

- Query projections
- Key projections
- Value projections

### LoRA configuration

```text
Rank (r):          32
Alpha:             64
Dropout:           0.05
Bias:              none
Target modules:    query, key, value
```


The resulting model contains:
```text
Trainable parameters: 2,419,277
Total parameters:     127,124,122
Trainable:            1.9031%
```

This means that approximately **1.9% of the model parameters are trainable** during the LoRA fine-tuning process.

The model uses a maximum sequence length of **256 tokens** during prediction.

---
## 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Macro F1
- Weighted F1
- Per-intent precision
- Per-intent recall
- Per-intent F1

### Overall Results

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| MLP baseline | 87.99% | 87.97% | 87.97% |
| **RoBERTa + LoRA** | **93.47%** | **93.47%** | **93.47%** |

### Key Result

The RoBERTa + LoRA model improved macro F1 by **5.48 percentage points** compared with the MLP baseline.

Macro F1 was used as the primary comparison metric because it gives equal weight to each of the 77 intent classes.

---

## 🔍 Intent-Level Performance

The project also compares F1 scores for individual banking intents.

Some of the largest improvements were observed for card-related and identity-verification intents.

| Intent | MLP F1 | RoBERTa + LoRA F1 |
| --- | ---: | ---: |
| `contactless_not_working` | 57.6% | **96.1%** |
| `card_not_working` | 66.7% | **91.6%** |
| `verify_my_identity` | 71.2% | **94.0%** |
| `unable_to_verify_identity` | 81.5% | **96.1%** |
| `why_verify_identity` | 80.0% | **96.3%** |

The analysis also identified **8 intents with small performance losses of less than 3%** for the RoBERTa + LoRA model. Both models still achieved F1 scores of at least 85% for those intents.

---

## 🔐 PII Protection

The project includes a PII-processing workflow before model prediction.

The implemented function can detect and process several types of sensitive information.

### Redaction

The following types of information can be replaced with redaction markers:

- Credit-card numbers
- CVV numbers
- Account numbers
- Routing/account numbers
- SSNs
- PINs
- Passport numbers
- Driver's license numbers
- Dates of birth
- Addresses
- Names
- ZIP codes
- IP addresses

### Hashing

Email addresses and phone numbers can either be redacted or hashed.

When hashing is enabled, **SHA-256** is used and the resulting hash is truncated to 16 hexadecimal characters.

### Example

```text
Original input:

I want to dispute a charge of $49.99 on my card
4532-1234-5678-9010 and CVV 000.
My email is john.doe@email.com,
call me at 123-456-7890.

                ↓

PII processing

                ↓

Redacted input:

I want to dispute a charge of $49.99 on my card
[CARD NUMBER REDACTED] and [CVV NUMBER REDACTED].
My email is [HASH:...],
call me at [HASH:...].
```

The cleaned text is then tokenized and passed to the classifier.

---

## 🧪 Example Prediction

The prediction function returns the top-k predicted banking intents.

For an example query concerning an unexpected charge, the model produced:

| Rank | Intent | Confidence |
| ---: | --- | ---: |
| 1 | `direct_debit_payment_not_recognised` | 60.5% |
| 2 | `extra_charge_on_statement` | 32.4% |
| 3 | `transaction_charged_twice` | 2.9% |

The example also illustrates the difficulty of distinguishing between semantically similar banking intents.

---

## ⚙️ Key Engineering Challenges
### Multi-class intent classification

The task involves **77 different banking intents**, including several categories with similar terminology and meanings.

### Comparing feature-based and transformer approaches

The project compares a traditional TF-IDF representation with a pretrained transformer-based representation, providing a direct comparison between the MLP baseline and RoBERTa + LoRA.

### Parameter-efficient fine-tuning

LoRA was used to adapt RoBERTa while keeping the number of trainable parameters to approximately 1.9% of the full model.

### Sensitive information handling

The prediction workflow processes sensitive customer information before classification through PII redaction and optional hashing.

### Intent-level evaluation

Beyond overall accuracy and F1 scores, the project examines performance for individual intents to identify where the models improve or lose performance.

---

## 🧠 Technical Skills Demonstrated
- Natural Language Processing (NLP)
- Multi-Class Text Classification
- TF-IDF Feature Engineering
- Neural Network Development
- Transformer Fine-Tuning
- Parameter-Efficient Fine-Tuning (LoRA)
- Model Evaluation & Comparison
- Per-Class Performance Analysis
- PII Detection & Data Protection
- Data Visualization
- Python Machine Learning Development

---

## 💡 Key Insights
### Transformer-based classification improved overall performance

RoBERTa + LoRA achieved **93.47% macro F1**, compared with **87.97%** for the MLP baseline.

### Improvements were visible at the intent level

Several card-related and identity-verification intents showed substantial F1 improvements when using RoBERTa + LoRA.

### LoRA reduced the number of trainable parameters

Only approximately **1.9%** of the total RoBERTa parameters were trainable during fine-tuning.

### PII processing can be integrated into inference

The project demonstrates a prediction workflow in which sensitive information is processed before tokenization and classification.

---

## 🚀 Future Steps

Potential extensions identified in the project include:

### Model Improvements
- Ensemble methods combining predictions from multiple models
- Hyperparameter optimization
- Experimenting with different LoRA configurations
- Experimenting with different learning rates
- Testing larger transformer models
- Human-in-the-loop handling for low-confidence predictions
- Active learning using uncertain predictions

### Inference & Monitoring
- Performance monitoring
- Data-drift monitoring
- Inference-latency optimization
- Model quantization
- Caching strategies
- Feedback loops for future model improvement

---

## 📦 Technologies
- Python
- PyTorch
- Scikit-learn
- Hugging Face Transformers
- RoBERTa
- PEFT / LoRA
- Pandas
- NumPy
- Matplotlib
- WordCloud
- Git

---

## 📁 Project Structure
```text
.
├── README.md
└── banking_issues_classification.ipynb
```

---

## 📌 Summary
This project compares a **TF-IDF + MLP** baseline with **RoBERTa + LoRA** for multi-class banking intent classification.

The RoBERTa + LoRA approach achieved:
- 93.47% accuracy
- 93.47% macro F1
- 93.47% weighted F1
- +5.48 percentage points macro F1 over the MLP baseline

The project additionally incorporates **PII protection into the prediction workflow**, including sensitive-data redaction and optional hashing of email and phone identifiers.
