"""
Banking Support AI — RAG Streamlit Demo (chat interface)
Pipeline:
  1. Fine-tuned BERT       -> classifies the query into 1 of 77 intents
  2. Sentence-Transformers
     + FAISS               -> retrieves the most relevant knowledge doc(s)
  3. Plain TinyLlama (base model, no fine-tuning) -> generates a response
     grounded in that retrieved context

Separate variant from the QLoRA project -- tests RAG on its own, not combined
with the QLoRA adapter, so the two techniques can be compared independently.

Folder layout expected (this file sits alongside the BERT model folder):
    Banking Support AI/
    ├── app_rag.py
    └── banking77-bert-final/

Run with:
    pip install streamlit transformers torch accelerate safetensors sentence-transformers faiss-cpu
    streamlit run app_rag.py
"""

import streamlit as st
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

BERT_MODEL_PATH = "banking77-bert-final"
BASE_LLM_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MIN_RETRIEVAL_SCORE = 0.35
LOW_CONFIDENCE_THRESHOLD = 0.50

EXAMPLE_QUERIES = [
    "My card still hasn't arrived after two weeks 📦",
    "Why was my transfer declined? 🚫",
    "I don't recognize a cash withdrawal on my account 💸",
    "How do I verify my identity? 🪪",
]

# A small emoji per intent, purely cosmetic -- makes the predicted-intent line
# a bit more fun to glance at without changing anything about the pipeline.
INTENT_EMOJI = {
    "card_arrival": "📦", "card_not_working": "💳", "declined_card_payment": "🚫",
    "lost_or_stolen_card": "🔒", "change_pin": "🔢", "activate_my_card": "✅",
    "pending_transfer": "⏳", "failed_transfer": "❌", "declined_transfer": "🚫",
    "Refund_not_showing_up": "↩️", "cash_withdrawal_not_recognised": "💸",
    "exchange_rate": "💱", "top_up_by_card_charge": "➕", "wrong_amount_of_cash_received": "🏧",
    "verify_my_identity": "🪪",
}

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


@st.cache_resource(show_spinner="Loading BERT intent classifier...")
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Building the retrieval index...")
def build_retrieval_index():
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    documents = [doc["content"] for doc in KNOWLEDGE_BASE]
    embeddings = embedder.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return embedder, index


@st.cache_resource(show_spinner="Waking up the response generator (first load can take a minute on CPU)...")
def load_generator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(BASE_LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(BASE_LLM_NAME, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def classify_intent(query, tokenizer, model):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    predicted_id = torch.argmax(probs).item()
    return model.config.id2label[predicted_id], probs[predicted_id].item()


def retrieve_documents(query, embedder, index, top_k=2):
    query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_embedding, top_k)
    return [{**KNOWLEDGE_BASE[idx], "score": float(score)} for score, idx in zip(scores[0], indices[0])]


def clean_response(text: str) -> str:
    for marker in ["Answer:", "The final answer is:", "The answer is:"]:
        if marker in text:
            return text.split(marker, 1)[-1].strip()
    return text


def generate_rag_response(query, retrieved_docs, tokenizer, model, device, max_new_tokens=150):
    context = "\n\n".join(f"Document: {d['title']}\n{d['content']}" for d in retrieved_docs)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a banking support assistant. Answer the customer's question using "
                "the provided knowledge context. Do not invent bank policies, fees, deadlines, "
                "balances, or requirements. Do not repeat, copy, or quote the knowledge context "
                "or the customer's question back. Respond with only your direct advice to the "
                "customer, in 2-3 short sentences."
            ),
        },
        {"role": "user", "content": f"Knowledge context:\n{context}\n\nCustomer question: {query}"},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Priming: ending the prompt with "Answer:" ourselves forces generation to
    # continue straight into an answer, instead of leaving the model free to
    # restate the context first (which it does often enough to matter) and
    # potentially run out of its token budget before ever answering.
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


def confidence_badge(confidence: float) -> str:
    if confidence >= 0.75:
        return f":green[●] **{confidence:.1%} confidence**"
    if confidence >= LOW_CONFIDENCE_THRESHOLD:
        return f":orange[●] **{confidence:.1%} confidence**"
    return f":red[●] **{confidence:.1%} confidence**"


def run_pipeline(query: str):
    bert_tokenizer, bert_model = load_classifier()
    intent, confidence = classify_intent(query, bert_tokenizer, bert_model)
    emoji = INTENT_EMOJI.get(intent, "🏦")

    embedder, index = build_retrieval_index()
    retrieved = retrieve_documents(query, embedder, index, top_k=2)

    with st.chat_message("assistant"):
        st.markdown(f"**{emoji} Intent:** `{intent}`  —  {confidence_badge(confidence)}")
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            st.warning("Low confidence — in a real deployment this would route to a human agent.")

        with st.expander("🔍 Show retrieved knowledge (why I'm answering this way)"):
            for doc in retrieved:
                st.progress(min(doc["score"], 1.0), text=f"{doc['title']} — similarity {doc['score']:.3f}")

        if retrieved[0]["score"] < MIN_RETRIEVAL_SCORE:
            answer = (
                f"I don't have a confident match in my knowledge base for this "
                f"(best similarity {retrieved[0]['score']:.3f}). I'd rather escalate this to a "
                f"human agent than guess. 🙋"
            )
            st.error(answer)
        else:
            with st.spinner("Thinking... 💭"):
                llm_tokenizer, llm_model, device = load_generator()
                answer = generate_rag_response(query, retrieved, llm_tokenizer, llm_model, device)
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Banking Support AI", page_icon="🏦", layout="centered")

with st.sidebar:
    st.header("🏦 How this works")
    st.markdown(
        "1. **BERT** classifies your message into 1 of 77 banking intents\n"
        "2. **FAISS + Sentence-Transformers** retrieve the most relevant policy snippet\n"
        "3. **TinyLlama** generates a reply grounded in that snippet"
    )
    st.divider()
    st.caption("Running on a small (1.1B) model with no GPU, so replies may take a little while. 🐢")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🏦 Banking Support AI")
st.caption("Ask me a banking question — I'll figure out what you need and look it up before answering.")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.write("**Try one:**")
cols = st.columns(2)
for i, example in enumerate(EXAMPLE_QUERIES):
    if cols[i % 2].button(example, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": example})
        st.session_state.pending_query = example

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

typed_query = st.chat_input("Type your banking question...")
pending = st.session_state.pop("pending_query", None)
query_to_run = pending or typed_query

if typed_query:
    st.session_state.messages.append({"role": "user", "content": typed_query})
    with st.chat_message("user"):
        st.write(typed_query)

if query_to_run:
    run_pipeline(query_to_run)