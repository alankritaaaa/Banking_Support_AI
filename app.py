"""
Banking Support AI — Streamlit Demo
Chains together:
  1. Fine-tuned BERT -> classifies the customer query into 1 of 77 intents
  2. QLoRA-tuned TinyLlama -> generates a helpful support response

Folder layout expected (app.py sits alongside both model folders):
    Banking Support AI/
    ├── app.py                          <- this file
    ├── banking77-bert-final/           <- your saved BERT classifier
    └── banking-support-qlora-final/    <- your saved LoRA adapter

Run with:
    pip install streamlit transformers peft torch accelerate safetensors
    streamlit run app.py
"""

import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from peft import PeftModel

BERT_MODEL_PATH = "banking77-bert-final"
LORA_ADAPTER_PATH = "banking-support-qlora-final"
BASE_LLM_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Same 15 intents the generator was actually fine-tuned on (from your notebook).
# The BERT classifier knows all 77 -- the generator only knows how to respond
# well for these. Being upfront about this gap in the UI matters.
GENERATOR_SUPPORTED_INTENTS = {
    "card_arrival", "card_not_working", "declined_card_payment",
    "lost_or_stolen_card", "change_pin", "activate_my_card",
    "pending_transfer", "failed_transfer", "declined_transfer",
    "Refund_not_showing_up", "cash_withdrawal_not_recognised",
    "exchange_rate", "top_up_by_card_charge", "wrong_amount_of_cash_received",
    "verify_my_identity",
}


@st.cache_resource(show_spinner="Loading BERT intent classifier...")
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading response generator (this can take a minute on CPU)...")
def load_generator():
    # No 4-bit quantization here -- that needs a CUDA GPU. For a local/CPU demo,
    # load the base model at normal precision and apply the LoRA adapter on top;
    # the adapter itself is precision-agnostic, it was just TRAINED under
    # 4-bit quantization to save memory during fine-tuning.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_LLM_NAME, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def classify_intent(query: str, tokenizer, model):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    predicted_id = torch.argmax(probs).item()
    predicted_intent = model.config.id2label[predicted_id]
    confidence = probs[predicted_id].item()
    return predicted_intent, confidence


def generate_response(query: str, intent: str, tokenizer, model, device, max_new_tokens=80):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful banking support assistant. "
                "Give concise, clear, and relevant support responses. "
                "Do not invent policies, fees, deadlines, or requirements."
            ),
        },
        {
            "role": "user",
            "content": f"Customer intent: {intent}\nCustomer query: {query}",
        },
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return full_text[len(prompt_text):].strip()


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Banking Support AI", page_icon="🏦")
st.title("🏦 Banking Support AI")
st.caption("BERT classifies the intent -> a QLoRA-fine-tuned TinyLlama generates the response")

query = st.text_area(
    "Customer query",
    placeholder="e.g. My card still hasn't arrived after two weeks",
    height=100,
)

if st.button("Analyze & Respond", type="primary"):
    if not query.strip():
        st.warning("Please enter a customer query first.")
    else:
        bert_tokenizer, bert_model = load_classifier()
        intent, confidence = classify_intent(query, bert_tokenizer, bert_model)

        st.subheader("1. Predicted Intent")
        st.write(f"**{intent}**  ({confidence:.1%} confidence)")

        if intent not in GENERATOR_SUPPORTED_INTENTS:
            st.info(
                f"Note: the response generator was fine-tuned on 15 representative intents, "
                f"and `{intent}` isn't one of them. Generating with the base model's general "
                f"capability rather than fine-tuned behavior for this specific intent."
            )

        llm_tokenizer, llm_model, device = load_generator()
        with st.spinner("Generating response..."):
            response = generate_response(query, intent, llm_tokenizer, llm_model, device)

        st.subheader("2. Generated Response")
        st.write(response)

st.divider()
st.caption(
    "Note: TinyLlama-1.1B is a small model used to keep this demo runnable without a paid GPU. "
    "Response fluency reflects that model size, not the fine-tuning approach itself."
)
