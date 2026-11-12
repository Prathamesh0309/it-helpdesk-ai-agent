import asyncio
import json
import os
import sys

from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import GENERATION_MODEL, generate
from observability import annotate, describe_text, span, trace
from phase1_rag import format_history, generate_answer, is_relevant, retrieve

SERVER_PARAMS = StdioServerParameters(command=sys.executable, args=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")])

# Constrained decoding: the model can only return JSON matching this schema, so
# there is no regex-scraping of free text and no malformed-JSON crash.
DECISION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "needs_ticket": types.Schema(type=types.Type.BOOLEAN),
        "ticket_priority": types.Schema(type=types.Type.STRING, enum=["low", "medium", "high"]),
        "needs_slack": types.Schema(type=types.Type.BOOLEAN),
        "slack_urgency": types.Schema(type=types.Type.STRING, enum=["normal", "urgent"]),
        "ticket_id_to_check": types.Schema(
            type=types.Type.STRING,
            nullable=True,
            description="Ticket ID (e.g. TKT-A3X9F1) if the employee asks about an existing ticket, else null",
        ),
        "reasoning": types.Schema(type=types.Type.STRING),
    },
    required=["needs_ticket", "needs_slack", "reasoning"],
)

SAFE_DEFAULT = {
    "needs_ticket": False,
    "needs_slack": False,
    "ticket_id_to_check": None,
    "reasoning": "Could not parse decision; taking no action.",
}


def decide_actions(query, rag_answer, history):
    """Ask the LLM which tools (if any) to call. Returns a dict."""
    prompt = f"""You are an IT Support Orchestrator Agent.

CONVERSATION HISTORY:
{format_history(history)}

You have already retrieved this answer from the knowledge base:
RAG ANSWER: {rag_answer}

Based on the conversation history and the employee's latest question, decide which actions to take.
Tools:
1. create_ticket - the issue needs tracking or the answer didn't fully resolve it
2. notify_slack - the issue is urgent or needs immediate IT attention
3. check_ticket_status - the employee asks about an existing ticket (set ticket_id_to_check)

Do not create a ticket for a simple question the answer fully resolves.

EMPLOYEE LATEST QUESTION: {query}"""

    try:
        response = generate(
            GENERATION_MODEL,
            prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DECISION_SCHEMA,
                temperature=0,  # routing should be deterministic
            ),
        )
        return {**SAFE_DEFAULT, **json.loads(response.text)}
    except Exception as e:  # network error, blocked response, bad JSON...
        print(f"  ! decision step failed: {e}", file=sys.stderr)
        return dict(SAFE_DEFAULT)


async def call_tool(session, name, args):
    with span(f"tool.{name}"):
        result = await session.call_tool(name, args)
    return result.content[0].text


async def orchestrate(session, query, employee_name, history):
    """Handle one employee message. Each call is one trace (see observability.py)."""
    with trace("support_request", query=describe_text(query), turn=len(history) // 2 + 1):
        return await _orchestrate(session, query, employee_name, history)


async def _orchestrate(session, query, employee_name, history):
    print(f"\n{'='*55}\n  Query: {query}\n{'='*55}")

    # Step 1: RAG (retrieval sees history via query rewriting)
    print("\n[1/3] Running RAG pipeline...")
    chunks = await asyncio.to_thread(retrieve, query, history)
    rag_answer = await asyncio.to_thread(generate_answer, query, chunks, history)
    print(f"  ✓ RAG answer generated from {len(chunks)} chunks")
    annotate(answered_from_kb=is_relevant(chunks))

    # Step 2: Reasoning
    print("\n[2/3] Reasoning about actions...")
    with span("decide"):
        decision = await asyncio.to_thread(decide_actions, query, rag_answer, history)

    # Deterministic rule on top of the LLM: if the knowledge base had nothing
    # relevant, a human has to pick this up, so always open a ticket.
    if not is_relevant(chunks) and not decision.get("ticket_id_to_check"):
        decision["needs_ticket"] = True
        decision.setdefault("ticket_priority", "medium")
        decision["reasoning"] += " (Escalated: no relevant knowledge-base content.)"
    print(f"  ✓ Decision: {decision['reasoning']}")
    annotate(escalated=bool(decision.get("needs_ticket")), notified_slack=bool(decision.get("needs_slack")))

    # Step 3: Act
    print("\n[3/3] Taking actions...")
    actions_taken = []

    if decision.get("ticket_id_to_check"):
        print("  → Checking ticket status...")
        result = await call_tool(session, "check_ticket_status",
                                 {"ticket_id": decision["ticket_id_to_check"]})
        actions_taken.append(f"Ticket status: {result}")

    if decision.get("needs_ticket"):
        print("  → Creating ticket...")
        result = await call_tool(session, "create_ticket", {
            "employee_name": employee_name,
            "issue_summary": query,
            "priority": decision.get("ticket_priority", "medium"),
        })
        actions_taken.append(f"Ticket created: {result}")

    if decision.get("needs_slack"):
        print("  → Notifying Slack...")
        result = await call_tool(session, "notify_slack", {
            "channel": "it-help",
            "message": f"Employee {employee_name} needs help: {query}",
            "urgency": decision.get("slack_urgency", "normal"),
        })
        actions_taken.append(f"Slack notified: {result}")

    if not actions_taken:
        print("  ✓ No actions needed - RAG answer is sufficient")

    final_response = rag_answer
    if actions_taken:
        final_response += "\n\n--- Actions Taken ---"
        for action in actions_taken:
            final_response += f"\n• {action}"
    return final_response


async def main():
    print("=== IT Support Copilot - Orchestrator ===")
    print("Type 'quit' to exit\n")

    employee = (await asyncio.to_thread(input, "Your name: ")).strip() or "Anonymous"
    print(f"\nHello {employee}! How can I help you today?")
    print("(Type 'reset' to start a new conversation)\n")

    history = []

    # One MCP server process and session for the whole conversation, rather
    # than spawning a new subprocess for every tool call.
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            while True:
                question = (await asyncio.to_thread(input, "You: ")).strip()

                if question.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break
                if question.lower() == "reset":
                    history.clear()
                    print("Conversation reset! Starting fresh.\n")
                    continue
                if not question:
                    continue

                try:
                    answer = await orchestrate(session, question, employee, history)
                except Exception as e:
                    print(f"\nSomething went wrong: {e}\n")
                    continue

                # Append AFTER answering so the current turn isn't in its own history.
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": answer})
                print(f"\n🤖 IT Copilot:\n{answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
