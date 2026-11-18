"""Banking Support AI core: your existing logic from rag_app.py, with Streamlit removed.

Three plain functions (used by server.py as MCP tools):
    classify_intent(text)           -> {"intent", "confidence"}      (fine-tuned BERT)
    search_policy(query)            -> {"documents": [...], "score"} (MiniLM + FAISS)
    generate_reply(text, documents) -> str                           (base TinyLlama, RAG prompt)

Models load lazily on first use and are cached. Never print() in this file:
MCP's stdio transport uses stdout, so a stray print breaks the connection.
"""
from functools import lru_cache
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

# Resolved relative to this file, so it works no matter where the server is launched from.
BERT_MODEL_PATH = str(Path(__file__).parent / "banking77-bert-final")
BASE_LLM_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Which model writes the final reply?
#   "ollama"    -> your local Ollama model (follows "don't invent" far better than TinyLlama)
#   "tinyllama" -> the original base TinyLlama from rag_app.py
GENERATOR = "ollama"
OLLAMA_MODEL = "llama3.1"  # already pulled on your laptop; if too slow try: ollama pull llama3.2  (and set "llama3.2")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

KNOWLEDGE_BASE = [
    {"title": "Card arrival", "intent": "card_arrival", "content": "If a customer is waiting for a physical bank card, they can check the card delivery status through the bank's official app or support channel. If the expected delivery period has passed, the customer should contact the bank through an official support channel. Do not assume a specific delivery deadline unless the bank has provided one."},
    {"title": "Card not working", "intent": "card_not_working", "content": "If a bank card is not working, the customer can check whether the card is active, whether it has expired, and whether the issue occurs at multiple merchants or terminals. If the card continues to fail, the customer should contact the bank through its official support channel."},
    {"title": "Declined card payment", "intent": "declined_card_payment", "content": "A card payment can be declined for different reasons, including insufficient available funds, security checks, merchant issues, card restrictions, or technical problems. The customer should check the transaction details in the banking app and contact the bank if the reason is unclear."},
    {"title": "Lost or stolen card", "intent": "lost_or_stolen_card", "content": "If a customer has lost a bank card or believes it was stolen, they should immediately use the bank's official app or contact the bank through an official support channel to freeze or block the card. Customers should not share their PIN, password, or OTP with anyone."},
    {"title": "Change PIN", "intent": "change_pin", "content": "Customers should use the bank's official app, ATM, or other officially supported method to change their card PIN. The exact process depends on the bank. Customers should never share their PIN with support staff or other people."},
    {"title": "Activate card", "intent": "activate_my_card", "content": "A new bank card may need to be activated before it can be used. Customers should follow the activation instructions provided by their bank, such as using the official banking app or another officially supported method."},
    {"title": "Pending transfer", "intent": "pending_transfer", "content": "A transfer marked as pending has not yet reached a completed state. Customers should check the transaction status in their banking app. If the transfer remains pending longer than the bank's stated processing period, they should contact official bank support."},
    {"title": "Failed transfer", "intent": "failed_transfer", "content": "A failed transfer means the transaction was not completed. Customers should review the transaction details and any error message shown by the bank. If the reason is unclear, they should contact the bank through an official support channel."},
    {"title": "Declined transfer", "intent": "declined_transfer", "content": "A transfer may be declined because of account restrictions, security checks, insufficient available funds, incorrect details, or technical issues. Customers should review the transaction information and contact official bank support if they need further clarification."},
    {"title": "Refund not showing", "intent": "Refund_not_showing_up", "content": "If a customer is expecting a refund that has not appeared, they should first check the original transaction and contact the merchant if necessary. The customer can also check the bank transaction history. Exact refund processing times depend on the merchant, payment method, and bank, so a specific deadline should not be assumed without verified information."},
    {"title": "Unrecognized cash withdrawal", "intent": "cash_withdrawal_not_recognised", "content": "If a customer does not recognize a cash withdrawal, they should review the transaction details and immediately contact their bank through an official support channel. If the card may be compromised, they should follow the bank's official procedure for securing or blocking the card."},
    {"title": "Exchange rate", "intent": "exchange_rate", "content": "Exchange rates can change over time and may differ between providers and transaction types. Customers should check the current exchange rate shown by their bank or official banking service before making a transaction."},
    {"title": "Top up by card charge", "intent": "top_up_by_card_charge", "content": "A card top-up may involve charges depending on the bank, card, payment method, or account type. Customers should check the fee information provided by their bank before assuming that a particular fee applies."},
    {"title": "Wrong cash amount", "intent": "wrong_amount_of_cash_received", "content": "If an ATM dispenses an incorrect amount of cash, the customer should keep the transaction details and contact the bank through its official support channel as soon as possible. They should not assume a particular resolution time without information from the bank."},
    {"title": "Identity verification", "intent": "verify_my_identity", "content": "Identity verification may be required for certain banking services. Customers should complete verification only through the bank's official app, website, or another officially provided channel. Sensitive documents and personal information should not be shared through unofficial channels."},
]


@lru_cache(maxsize=1)
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def build_retrieval_index():
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    documents = [doc["content"] for doc in KNOWLEDGE_BASE]
    embeddings = embedder.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return embedder, index


@lru_cache(maxsize=1)
def load_generator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(BASE_LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(BASE_LLM_NAME, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def clean_response(text: str) -> str:
    for marker in ["Answer:", "The final answer is:", "The answer is:"]:
        if marker in text:
            return text.split(marker, 1)[-1].strip()
    return text


SYSTEM_PROMPT = (
    "You are a banking support assistant. Answer the customer's question using "
    "the provided knowledge context. Do not invent bank policies, fees, deadlines, "
    "balances, or requirements. Do not repeat, copy, or quote the knowledge context "
    "or the customer's question back. Respond with only your direct advice to the "
    "customer, in 2-3 short sentences."
)


def generate_with_ollama(query, retrieved_docs):
    """Reply via the local Ollama server. Returns "" if Ollama is not reachable (the guardrail then falls back)."""
    import ollama
    context = "\n\n".join(f"Document: {d['title']}\n{d['content']}" for d in retrieved_docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Knowledge context:\n{context}\n\nCustomer question: {query}"},
    ]
    try:
        resp = ollama.chat(model=OLLAMA_MODEL, messages=messages,
                           options={"temperature": 0.2, "num_predict": 120})
        return resp["message"]["content"].strip()
    except Exception:
        return ""


def generate_rag_response(query, retrieved_docs, tokenizer, model, device, max_new_tokens=150):
    context = "\n\n".join(f"Document: {d['title']}\n{d['content']}" for d in retrieved_docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Knowledge context:\n{context}\n\nCustomer question: {query}"},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_text += "Answer:"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_k=40,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return clean_response(raw_response)


# ---------------- The 3 functions exposed through MCP ----------------
def classify_intent(text: str) -> dict:
    tokenizer, model = load_classifier()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    predicted_id = torch.argmax(probs).item()
    return {"intent": model.config.id2label[predicted_id], "confidence": probs[predicted_id].item()}


def search_policy(query: str, top_k: int = 2) -> dict:
    embedder, index = build_retrieval_index()
    query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_embedding, top_k)
    docs = [{**KNOWLEDGE_BASE[idx], "score": float(score)} for score, idx in zip(scores[0], indices[0])]
    return {"documents": docs, "score": docs[0]["score"]}


def generate_reply(text: str, documents: list) -> str:
    if GENERATOR == "ollama":
        return generate_with_ollama(text, documents)
    tokenizer, model, device = load_generator()
    return generate_rag_response(text, documents, tokenizer, model, device, max_new_tokens=100)
