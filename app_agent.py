"""Streamlit chat UI for the Banking Support agent (LangGraph + MCP + guardrails).

Run (from this folder):   streamlit run app_agent.py
It reuses agent_graph.py, so every setting there (USE_LLM, thresholds, ...) applies here too.
Your original app_rag.py is untouched.
"""
import asyncio
import sys
import threading

import streamlit as st
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

import agent_graph as ag

EXAMPLE_QUERIES = [
    "My card still hasn't arrived after two weeks 📦",
    "Why was my transfer declined? 🚫",
    "I don't recognize a cash withdrawal on my account 💸",
    "What is the weather today? ☁️",
]
STATUS_LABEL = {
    "answered": ":green[● answered by the LLM, checked by the guardrail]",
    "fallback": ":orange[● LLM reply rejected by the guardrail, answered from the knowledge base]",
    "kb_only": ":blue[● answered from the knowledge base (LLM off)]",
    "escalated": ":red[● escalated to a human agent]",
}


@st.cache_resource(show_spinner="Starting the banking agent (MCP server + models)... first start can take a minute.")
def start_agent():
    """Run the MCP session and the LangGraph app in ONE background event loop that stays alive,
    so the models load once and every chat message reuses them."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    holder = {}

    async def run():
        client = MultiServerMCPClient(
            {"banking": {"command": sys.executable, "args": ["server.py"], "transport": "stdio"}}
        )
        async with client.session("banking") as session:
            tools = {t.name: t for t in await load_mcp_tools(session)}
            holder["app"] = ag.build_graph(tools)
            ready.set()
            await asyncio.Event().wait()  # keep the session (and the server process) open

    def runner():
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run())
        except Exception as e:  # report start-up failures to the page instead of hanging
            holder["error"] = e
            ready.set()

    threading.Thread(target=runner, daemon=True).start()
    ready.wait(timeout=180)
    if "error" in holder:
        raise RuntimeError(f"Agent failed to start: {holder['error']!r}")
    if "app" not in holder:
        raise RuntimeError("Agent did not start within 3 minutes. Try running `py server.py` to see the error.")
    return loop, holder["app"]


def ask(question: str) -> dict:
    loop, app = start_agent()
    return asyncio.run_coroutine_threadsafe(app.ainvoke({"question": question}), loop).result()


def render_assistant(msg: dict):
    meta = msg.get("meta")
    if meta:
        st.markdown(f"**Intent:** `{meta['intent']}` ({meta['confidence']:.0%})  ·  {STATUS_LABEL[meta['status']]}")
    st.write(msg["content"])
    if meta:
        with st.expander("🔍 How this answer was produced"):
            st.write(f"Best retrieval similarity: **{meta['score']:.2f}** "
                     f"(needs ≥ {ag.MIN_RETRIEVAL_SCORE}); intent confidence needs ≥ {ag.LOW_CONFIDENCE_THRESHOLD}")
            for d in meta.get("documents", []):
                st.progress(max(0.0, min(d["score"], 1.0)), text=f"{d['title']} — similarity {d['score']:.3f}")
            if meta["status"] == "fallback":
                st.warning(f"Guardrail rejected the LLM reply: {meta['reason']}")
                st.caption(f"LLM said: {meta['raw_reply'][:400]}")


# ---------------- UI ----------------
st.set_page_config(page_title="Banking Support AI", page_icon="🏦", layout="centered")

with st.sidebar:
    st.header("🏦 How this works")
    st.markdown(
        "1. **BERT** classifies the intent (1 of 77)\n"
        "2. **FAISS + MiniLM** retrieve the best policy documents\n"
        "3. **LangGraph** decides: answer or escalate to a human\n"
        "4. **LLM** writes the reply, a **guardrail** checks it\n"
        "5. The three tools run on an **MCP server**"
    )
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🏦 Banking Support AI")
st.caption("LangGraph + MCP agent with an output guardrail.")

if "messages" not in st.session_state:
    st.session_state.messages = []

cols = st.columns(2)
for i, example in enumerate(EXAMPLE_QUERIES):
    if cols[i % 2].button(example, use_container_width=True):
        st.session_state.pending_query = example

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant(msg)
        else:
            st.write(msg["content"])

query = st.session_state.pop("pending_query", None) or st.chat_input("Type your banking question...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 💭"):
            out = ask(query)
        msg = {"role": "assistant", "content": out["reply"], "meta": {
            "intent": out["intent"], "confidence": out["confidence"], "status": out["status"],
            "score": out["score"], "documents": out.get("documents", []),
            "reason": out.get("reason", ""), "raw_reply": out.get("raw_reply", "")}}
        render_assistant(msg)
    st.session_state.messages.append(msg)
