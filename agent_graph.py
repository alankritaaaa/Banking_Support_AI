"""LangGraph workflow over the Banking Support AI, with its tools served through MCP.

Flow:  classify -> retrieve -> (confident enough?) -> answer -> check_reply -> done
                                      `-> escalate to a human
check_reply is the output guardrail. The LLM reply is rejected (and replaced by the retrieved
knowledge text) if it is empty, too long, contains made-up chatter, has [placeholders], invents
numbers, or is not grounded in the retrieved documents.
Retrieval/confidence thresholds are the same ones used in rag_app.py.
"""
import asyncio
import json
import re
import sys
from typing import TypedDict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import END, START, StateGraph

# ------------------------- SETTINGS (edit these, no other change needed) -------------------------
USE_LLM = True                   # False = skip TinyLlama completely: answer straight from the knowledge base
                                 #         (much faster, zero hallucination, less conversational)
MIN_RETRIEVAL_SCORE = 0.35       # from rag_app.py
LOW_CONFIDENCE_THRESHOLD = 0.50  # from rag_app.py
MAX_REPLY_WORDS = 80             # LLM replies longer than this are rejected
MIN_GROUNDED = 0.40              # share of the reply's content words that must come from the retrieved docs
                                 # (measured: bad TinyLlama replies scored 0.11-0.27, grounded ones 0.89+)
BAD_PHRASES = ["previous conversation", "we last spoke", "best regards", "kind regards", "sincerely", "as an ai"]
# --------------------------------------------------------------------------------------------------

_STOP = set("the a an and or of to in on for with that this is are be you your can will should if not "
            "it as at by from have has any they their we our may also".split())


def _words(text: str):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 4 and w not in _STOP]


def review_reply(reply: str, question: str, documents: list):
    """Return (reply_to_use, reason). reason is None when the LLM reply is acceptable."""
    reply = (reply or "").strip()
    if reply and reply[-1] not in ".!?":
        cut = max(reply.rfind("."), reply.rfind("!"), reply.rfind("?"))
        if cut > 0:
            reply = reply[:cut + 1]  # drop the unfinished last sentence caused by the token cap
    source = question + " " + " ".join(d["content"] for d in documents)
    low = reply.lower()
    body = re.sub(r"(^|\s)\d+[.)]\s", " ", reply)  # ignore list markers like "1)"

    if not reply:
        return reply, "empty reply"
    if len(reply.split()) > MAX_REPLY_WORDS:
        return reply, f"too long ({len(reply.split())} words, limit {MAX_REPLY_WORDS})"
    hits = [p for p in BAD_PHRASES if p in low]
    if hits:
        return reply, f"contains {hits}"
    if re.search(r"\[[^\]]*\]", reply):
        return reply, "contains a [placeholder]"
    invented = set(re.findall(r"\d+", body)) - set(re.findall(r"\d+", source))
    if invented:
        return reply, f"invents numbers {sorted(invented)}"
    words = _words(reply)
    src_words = set(_words(source))
    grounded = sum(w in src_words for w in words) / max(len(words), 1)
    if grounded < MIN_GROUNDED:
        return reply, f"not grounded in the documents ({grounded:.0%} of its words come from them, need {MIN_GROUNDED:.0%})"
    return reply, None


class State(TypedDict, total=False):
    question: str
    intent: str
    confidence: float
    documents: list
    score: float
    reply: str
    status: str  # "answered", "fallback" (guardrail replaced the LLM reply), "kb_only" (USE_LLM off) or "escalated"
    raw_reply: str   # the LLM reply the guardrail rejected (only set on fallback)
    reason: str      # why it was rejected


def parse(result):
    """MCP tool results arrive as text blocks; turn them back into Python."""
    if isinstance(result, list):
        result = result[0]["text"] if isinstance(result[0], dict) else result[0]
    try:
        return json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return result


def build_graph(tools: dict):
    async def classify(state: State):
        out = parse(await tools["classify_intent"].ainvoke({"text": state["question"]}))
        return {"intent": out["intent"], "confidence": out["confidence"]}

    async def retrieve(state: State):
        out = parse(await tools["search_policy"].ainvoke({"query": state["question"]}))
        return {"documents": out["documents"], "score": out["score"]}

    async def answer(state: State):
        if not USE_LLM:
            return {"reply": f"Here is our guidance on this: {state['documents'][0]['content']}",
                    "status": "kb_only"}
        reply = parse(await tools["generate_reply"].ainvoke(
            {"text": state["question"], "documents": state["documents"]}))
        return {"reply": reply, "status": "answered"}

    def check_reply(state: State):
        if state.get("status") == "kb_only":
            return {}
        reply, reason = review_reply(state.get("reply"), state["question"], state["documents"])
        if reason is None:
            return {"reply": reply}  # good reply (possibly trimmed to its last full sentence)
        top = state["documents"][0]
        return {"reply": f"Here is our guidance on this: {top['content']}", "status": "fallback",
                "raw_reply": state.get("reply") or "", "reason": reason}

    async def escalate(state: State):
        return {"reply": "I don't have a confident answer for this, so I'm passing it to a human agent. 🙋",
                "status": "escalated"}

    def decide(state: State):
        confident = state["confidence"] >= LOW_CONFIDENCE_THRESHOLD and state["score"] >= MIN_RETRIEVAL_SCORE
        return "answer" if confident else "escalate"

    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_node("check_reply", check_reply)
    g.add_node("escalate", escalate)
    g.add_edge(START, "classify")
    g.add_edge("classify", "retrieve")
    g.add_conditional_edges("retrieve", decide, {"answer": "answer", "escalate": "escalate"})
    g.add_edge("answer", "check_reply")
    g.add_edge("check_reply", END)
    g.add_edge("escalate", END)
    return g.compile()


async def main():
    client = MultiServerMCPClient(
        {"banking": {"command": sys.executable, "args": ["server.py"], "transport": "stdio"}}
    )
    # ONE session = ONE server process for the whole run, so the models load only once.
    async with client.session("banking") as session:
        tools = {t.name: t for t in await load_mcp_tools(session)}
        app = build_graph(tools)
        print("Banking Support agent ready. Type a question (or 'quit').")
        while True:
            q = (await asyncio.to_thread(input, "\nYou: ")).strip()
            if q.lower() in {"quit", "exit"}:
                break
            if not q:
                continue
            out = await app.ainvoke({"question": q})
            print(f"  intent={out['intent']} ({out['confidence']:.0%})  best similarity={out['score']:.2f}  "
                  f"status={out['status']}\nAgent: {out['reply']}")
            if out["status"] == "fallback":
                print(f"  [guardrail] LLM reply rejected: {out['reason']}\n  [guardrail] LLM said: {out['raw_reply'][:400]}")


if __name__ == "__main__":
    asyncio.run(main())
