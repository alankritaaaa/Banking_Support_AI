# 🏦 Banking Support AI

An end-to-end banking support AI project combining **BERT intent classification, QLoRA fine-tuning, Retrieval-Augmented Generation (RAG), and lightweight safety checks**.

The project explores two approaches for generating banking-support responses:

- **BERT + QLoRA TinyLlama**
- **BERT + RAG + TinyLlama**

---

## 🚀 Architecture

### QLoRA Pipeline

```text
Customer Query
      ↓
BERT Intent Classifier
      ↓
Intent + Confidence
      ↓
Intent Router
      ↓
TinyLlama 1.1B + QLoRA
      ↓
Banking Support Response
````

### RAG Pipeline

```text
Customer Query
      ↓
BERT Intent Classifier
      ↓
Sentence-Transformer Embeddings
      ↓
FAISS Retrieval
      ↓
Relevant Banking Knowledge
      ↓
TinyLlama 1.1B
      ↓
Grounded Response
```

---

## 🧠 Models

### BERT — Intent Classification

* Fine-tuned on the **Banking77** dataset
* Classifies queries into all **77 intents**
* Test Accuracy: **87.86%**
* Test Weighted F1: **86.82%**

BERT answers:

> "What is the customer asking about?"

### TinyLlama — Response Generation

Two approaches were explored:

**QLoRA:** TinyLlama 1.1B Chat fine-tuned with LoRA adapters and 4-bit NF4 quantization.

**RAG:** The base TinyLlama model receives relevant information retrieved from a banking-support knowledge base instead of using a fine-tuned response adapter.

---

## 🔎 RAG Implementation

The RAG pipeline uses:

* **Sentence Transformers** — `all-MiniLM-L6-v2`
* **FAISS** — vector similarity search
* A small banking-support knowledge base
* **TinyLlama 1.1B** — response generation

The system retrieves the top relevant documents for a customer query and provides them as context to the LLM.

A retrieval sanity test across 7 representative queries achieved:

**Top-3 Retrieval Accuracy: 100%**

> This is a prototype evaluation using a small knowledge base and test set.

---

## 🛡️ Safety Checks

The RAG application includes lightweight checks for:

* Low BERT intent confidence
* Low retrieval relevance
* Prompt-echo behavior from TinyLlama

When confidence or retrieval relevance is insufficient, the application can avoid presenting an unsupported response and recommend escalation.

These are prototype-level safeguards and are not a replacement for production banking security or human review.

---

## 📊 Results

### BERT

| Metric           | Result |
| ---------------- | -----: |
| Intent Classes   |     77 |
| Test Accuracy    | 90.10% |
| Test Weighted F1 | 89.92% |

### Example

**Query**

> My card payment was declined

**Predicted Intent**

`declined_card_payment`

**Confidence**

80.3%

The system then uses the predicted intent and the selected response-generation approach to produce a banking-support response.

---

## ⚙️ QLoRA Configuration

* Base Model: TinyLlama 1.1B Chat
* Quantization: 4-bit NF4
* LoRA Rank: 8
* LoRA Alpha: 16
* LoRA Dropout: 0.05
* Target Modules: `q_proj`, `v_proj`
* Training Epochs: 1
* Learning Rate: `2e-4`
* Max Sequence Length: 256
* Hardware: NVIDIA T4 GPU

The QLoRA prototype supports response generation for **15 selected banking intents**.

---

## 🛠️ Tech Stack

`Python` `PyTorch` `Hugging Face Transformers` `Hugging Face PEFT` `BERT` `TinyLlama` `LoRA / QLoRA` `Sentence Transformers` `FAISS` `Streamlit` `scikit-learn` `Pandas` `Google Colab` `NVIDIA T4 GPU`

---

## 📁 Repository Structure

```text
banking-support-ai/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── Banking_Support_AI_Final.ipynb
├── Banking_Support_RAG.ipynb
├── Banking_77_notebook
│
├── app.py
├── rag_app.py
│
├── banking77-bert-final/
└── banking-support-qlora-final/
```

---

## ▶️ Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the QLoRA application:

```bash
streamlit run app.py
```

Run the RAG application:

```bash
streamlit run rag_app.py
```

---

## ⚠️ Limitations

This is a **prototype banking-support system** and does not connect to real customer accounts, transactions, balances, or banking APIs.

The current RAG knowledge base contains general demonstration guidance rather than official bank policies.

TinyLlama 1.1B is also a relatively small language model, so response quality and instruction-following are limited compared with larger LLMs.

---

## 🔮 Future Improvements

* Larger and more authoritative knowledge bases
* Improved retrieval and reranking
* Automated RAG evaluation
* Stronger guardrails
* Tool calling and API integration
* Agent-based workflows
* Cloud deployment
* Production monitoring

---

## 🎯 Project Objective

This project demonstrates a progression from **NLP classification and fine-tuning to modern LLM application development**, exploring how BERT, QLoRA, RAG, vector retrieval, LLM generation, and application-level safety checks can work together in an AI support system.

```
