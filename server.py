"""MCP server: exposes the Banking Support AI pipeline as tools. No print() here (stdout = MCP channel)."""
from mcp.server.fastmcp import FastMCP

import banking_core

mcp = FastMCP("banking")


@mcp.tool()
def classify_intent(text: str) -> dict:
    """Classify a banking customer message into one of 77 intents; returns the intent and a confidence between 0 and 1."""
    return banking_core.classify_intent(text)


@mcp.tool()
def search_policy(query: str) -> dict:
    """Search the bank's knowledge base; returns the top matching documents and the best similarity score between 0 and 1."""
    return banking_core.search_policy(query)


@mcp.tool()
def generate_reply(text: str, documents: list[dict]) -> str:
    """Write a short customer-support reply to the message, grounded only in the given knowledge documents."""
    return banking_core.generate_reply(text, documents)


if __name__ == "__main__":
    mcp.run(transport="stdio")
