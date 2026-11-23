# 🏦 Banking Support AI

An end-to-end banking support AI prototype combining **BERT intent classification, QLoRA fine-tuning, Retrieval-Augmented Generation (RAG), LangGraph orchestration, MCP tool integration, and lightweight safety checks**.

The project explores three stages of banking-support response generation:

- **BERT + QLoRA TinyLlama**
- **BERT + RAG + TinyLlama**
- **BERT + RAG + LangGraph + MCP**

Each stage builds on the previous one, exploring different approaches to classification, knowledge retrieval, response generation, and agent orchestration.

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
```

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

### LangGraph + MCP Agent Pipeline

```text
Customer Query
      ↓
LangGraph Agent
      ↓
MCP Client
      ↓
MCP Server
      ↓
classify_intent
      ↓
search_policy
      ↓
Confidence + Relevance Check
      ↓
LLM Response Generation
      ↓
Output Guardrail
      ↓
Final Response / Escalation
```

---

## 🧠 Models

### BERT: Intent Classification

- Fine-tuned on the **Banking77** dataset
- Classifies queries into all **77 intents**
- Test Accuracy: **90.10%**
- Test Weighted F1: **89.92%**

BERT answers:

> "What is the customer asking about?"

### TinyLlama: Response Generation

Two approaches were explored:

**QLoRA:** TinyLlama 1.1B Chat fine-tuned with LoRA adapters and 4-bit NF4 quantization.

**RAG:** The base TinyLlama model receives relevant information retrieved from a banking-support knowledge base instead of using a fine-tuned response adapter.

The agent version also supports a local Ollama model (`llama3.1`) as an alternative response generator.

---

## 🔎 RAG Implementation

The RAG pipeline uses:

- **Sentence Transformers:** `all-MiniLM-L6-v2`
- **FAISS:** Vector similarity search
- A small banking-support knowledge base
- **TinyLlama 1.1B:** Response generation

The system retrieves the top relevant documents for a customer query and provides them as context to the LLM.

A retrieval sanity test across 7 representative queries achieved:

**Top-3 Retrieval Accuracy: 100%**

> This is a prototype evaluation using a small knowledge base and test set.

---

## 🛡️ Safety Checks

The RAG application includes lightweight checks for:

- Low BERT intent confidence
- Low retrieval relevance
- Prompt-echo behavior from TinyLlama

When confidence or retrieval relevance is insufficient, the application can avoid presenting an unsupported response and recommend escalation.

These are prototype-level safeguards and are not a replacement for production banking security or human review.

---

## 🤖 LangGraph + MCP Agent Layer

An agent layer built on top of the existing RAG pipeline.

It does **not retrain BERT or restructure the FAISS retrieval index**. Instead, it wraps the existing pipeline components as tools, orchestrates their execution using **LangGraph**, exposes them through an **MCP server**, and adds an output guardrail.

### Agent Pipeline

```text
Customer Query
      ↓
LangGraph Agent (MCP Client)
      ↓
Classify Intent
      ↓
Retrieve Banking Knowledge
      ↓
Confidence + Relevance Check
      ↓
   ┌─────────────────────────┐
   │                         │
 Below Threshold        Above Threshold
   │                         │
   ↓                         ↓
Human Escalation       LLM Response
                             ↓
                      Output Guardrail
                             ↓
                  ┌──────────┴──────────┐
                  │                     │
                Pass                  Fail
                  │                     │
                  ↓                     ↓
             Final Reply       Retrieved Knowledge
                               Fallback Response
```

### 🔌 MCP Server

The three pipeline components are exposed as tools through a local MCP server using **stdio transport**.

The LangGraph agent connects through an MCP client, discovers the available tools, and invokes them through the Model Context Protocol.

| MCP Tool | Responsibility | Technology |
|---|---|---|
| `classify_intent` | Predicts customer intent and confidence | Fine-tuned BERT |
| `search_policy` | Retrieves relevant documents and similarity scores | MiniLM + FAISS |
| `generate_reply` | Generates a response grounded in retrieved documents | Ollama / TinyLlama |

This separation makes the tools reusable by other MCP-compatible applications without embedding their implementation directly into the agent.

### 🔀 LangGraph Workflow

The agent workflow is implemented using:

- **State:** Maintains the customer query, predicted intent, confidence, retrieved documents, and generated reply.
- **Nodes:** `classify`, `retrieve`, `answer`, `check_reply`, and `escalate`.
- **Conditional Edges:** Routes queries based on intent confidence and retrieval relevance.
- **MCP Client:** Discovers and invokes tools exposed by the MCP server.

The existing thresholds are reused:

- BERT confidence threshold: `0.50`
- Minimum retrieval similarity: `0.35`

Queries below either threshold are escalated instead of being passed to the LLM. Off-topic queries are also escalated.

### 🛡️ Output Guardrail

Every generated LLM response is checked before being displayed.

The guardrail rejects a response and falls back to the retrieved knowledge text if it:

- Is empty or exceeds 80 words.
- Contains irrelevant or fabricated conversational content, such as "previous conversation" or "best regards".
- Contains placeholders or unsupported numbers.
- Fails the grounding check, where fewer than 40% of its content words appear in the retrieved documents.

Incomplete final sentences caused by the generation token limit are trimmed before validation.

The Streamlit interface displays whether a response was:

- Generated successfully by the LLM.
- Replaced by the output guardrail.
- Escalated due to insufficient confidence or retrieval relevance.

### 📊 Observations

- Base TinyLlama 1.1B ignored the instruction not to invent information in all 5 tested replies, adding unsupported phone numbers, business hours, and deadlines. The output guardrail rejected these responses.
- The local Ollama model (`llama3.1`) generated shorter responses that stayed within the retrieved information, with an observed response time of approximately 6–7 seconds per answer on a laptop.
- Off-topic questions, such as "What is the weather today?", were escalated in under a second.
- The grounding threshold was selected using a small sample: rejected TinyLlama responses scored between 0.11 and 0.27, while manually written grounded responses scored 0.89 or higher.

> These observations are based on a small prototype test set and are not a comprehensive benchmark.

---

## 📊 Results

### BERT

| Metric | Result |
|---|---:|
| Intent Classes | 77 |
| Test Accuracy | 90.10% |
| Test Weighted F1 | 89.92% |

### RAG Retrieval

| Metric | Result |
|---|---:|
| Test Queries | 7 |
| Top-3 Retrieval Accuracy | 100% |

### Example

**Query**

> My card payment was declined

**Predicted Intent**

`declined_card_payment`

**Confidence**

80.3%

The system uses the predicted intent and the selected response-generation approach to produce a banking-support response.

---

## ⚙️ QLoRA Configuration

- Base Model: TinyLlama 1.1B Chat
- Quantization: 4-bit NF4
- LoRA Rank: 8
- LoRA Alpha: 16
- LoRA Dropout: 0.05
- Target Modules: `q_proj`, `v_proj`
- Training Epochs: 1
- Learning Rate: `2e-4`
- Max Sequence Length: 256
- Hardware: NVIDIA T4 GPU

The QLoRA prototype supports response generation for **15 selected banking intents**.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Transformers](https://img.shields.io/badge/-Transformers-FF6F00?style=flat-square&logo=huggingface&logoColor=white)
![PEFT](https://img.shields.io/badge/-PEFT-6A5ACD?style=flat-square)
![BERT](https://img.shields.io/badge/-BERT-412991?style=flat-square)
![TinyLlama](https://img.shields.io/badge/-TinyLlama-8A2BE2?style=flat-square)
![LoRA](https://img.shields.io/badge/-LoRA-FF69B4?style=flat-square)
![RAG](https://img.shields.io/badge/-RAG-00A67E?style=flat-square)
![FAISS](https://img.shields.io/badge/-FAISS-0467DF?style=flat-square)
![Sentence Transformers](https://img.shields.io/badge/-Sentence%20Transformers-FFB000?style=flat-square)
![LangChain](https://img.shields.io/badge/-LangChain-1C3C3C?style=flat-square)
![LangGraph](https://img.shields.io/badge/-LangGraph-1C3C3C?style=flat-square)
![MCP](https://img.shields.io/badge/-MCP-6A5ACD?style=flat-square)
![Ollama](https://img.shields.io/badge/-Ollama-000000?style=flat-square)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Google Colab](https://img.shields.io/badge/-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![NVIDIA](https://img.shields.io/badge/-NVIDIA%20T4-76B900?style=flat-square&logo=nvidia&logoColor=white)
![LangGraph](https://img.shields.io/badge/-LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![LangChain](https://img.shields.io/badge/-LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![MCP](https://img.shields.io/badge/-MCP-000000?style=flat-square&logo=modelcontextprotocol&logoColor=white)
![Ollama](https://img.shields.io/badge/-Ollama-000000?style=flat-square&logo=ollama&logoColor=white)
![Llama 3.1](https://img.shields.io/badge/-Llama%203.1-0866FF?style=flat-square&logo=meta&logoColor=white)

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
├── agent_graph.py
├── server.py
├── banking_core.py
├── app_agent.py
│
├── banking77-bert-final/
└── banking-support-qlora-final/
```

### Agent Layer Files

| File | Responsibility |
|---|---|
| `agent_graph.py` | LangGraph workflow, MCP client, state management and output guardrail |
| `server.py` | MCP server exposing the three banking-support tools |
| `banking_core.py` | BERT classification, FAISS retrieval and response generation |
| `app_agent.py` | Streamlit interface for interacting with the agent |

---

## ▶️ Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

### Original QLoRA Application

```bash
streamlit run app.py
```

### RAG Application

```bash
streamlit run rag_app.py
```

### LangGraph + MCP Agent

Install and start Ollama, then pull the local model:

```bash
ollama pull llama3.1
```

Run the agent in the terminal:

```bash
python agent_graph.py
```

Run the agent Streamlit interface:

```bash
streamlit run app_agent.py
```

### Configuration

In `banking_core.py`:

- Set `GENERATOR = "ollama"` to use the local Ollama model.
- Set `GENERATOR = "tinyllama"` to use the original TinyLlama generator.

In `agent_graph.py`:

- Set `USE_LLM = False` to disable LLM generation and use retrieved knowledge as the response.

The MCP server is launched by the agent through the configured stdio connection.

---

## ⚠️ Limitations

This is a **prototype banking-support system**, not a production banking application.

- The knowledge base currently contains only 15 short demonstration documents.
- It does not connect to real customer accounts, balances, transactions, or authorised banking APIs.
- The output guardrail uses heuristic checks and a grounding threshold tuned on a small sample; it cannot guarantee that every hallucination will be detected.
- TinyLlama 1.1B has limited instruction-following capabilities.
- The Ollama generator was tested locally and may have different performance on other hardware.
- The current agent handles one customer question at a time and does not implement multi-turn conversation memory.
- Production use would require authentication, authorisation, secure data handling, comprehensive evaluation, monitoring, and human-review workflows.

---

## 🔮 Future Improvements

- Larger and more authoritative banking knowledge bases
- Improved retrieval and reranking
- Automated RAG and agent evaluation
- More robust output validation and guardrails
- Multi-turn conversation memory
- Integration with authorised banking APIs
- Human-in-the-loop approval workflows
- Multi-agent collaboration where appropriate
- Cloud deployment and production monitoring

---

## 🎯 Project Objective

This project demonstrates an evolution from traditional NLP classification to modern agentic AI application development.

It explores how **BERT, QLoRA, RAG, vector retrieval, LLM generation, LangGraph orchestration, MCP tool integration, and application-level safety checks** can work together to build a modular banking-support AI prototype.

---

If you've made it this far, thank you for checking out my project! 😄
I'm actively looking for opportunities in **AI/ML Engineering, NLP, and LLM application development**.

**HIRE ME !!!!**
