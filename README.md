# IT Support Agent

An agentic IT helpdesk assistant built on retrieval-augmented generation (RAG) and an MCP tool layer. It answers questions from a knowledge base, and when a question needs action it can open tickets and notify Slack through an MCP server.

## What it does

- **Retrieves** relevant support content from a local ChromaDB vector store using Gemini embeddings
- **Answers** with Gemini (`gemini-2.0-flash`), grounded in the retrieved context
- **Acts** through a FastMCP server exposing `create_ticket`, `check_ticket_status` and `notify_slack` (mock backends, designed to be swapped for real APIs)
- **Orchestrates** with a Plan → Act → Respond loop that decides whether to retrieve, call a tool, or answer
- **Filters tickets** in two layers: deterministic rules first, then an LLM-as-judge for ambiguous cases
- **Evaluates** answer quality against ground-truth test cases

<!-- Add an example session here: a user question, the tool calls made, and the final answer. -->

## Architecture

```
User question
    ↓
[Orchestrator]  Plan → Act → Respond
    ├── retrieve  → ChromaDB (Gemini embeddings)
    └── act       → MCP server: create_ticket / check_ticket_status / notify_slack
    ↓
Grounded answer
```

## Project structure

| File | Purpose |
| --- | --- |
| `phase1_ingest.py` | Loads `data/*.txt`, chunks content, embeds it, stores it in `db/` |
| `phase1_rag.py` | Query loop: retrieves top-K context and generates answers |
| `mcp_server.py` | MCP tool server (ticket creation, status, Slack notification) |
| `orchestrator.py` | Combines retrieval, reasoning and tool calls via MCP |
| `evalute_rag.py` | Scores RAG answers against ground-truth test cases |
| `data/` | Source knowledge documents (`.txt`) |
| `db/` | Local ChromaDB storage (generated) |

## Setup

Requires Python 3.10+ and a Gemini API key.

```bash
git clone https://github.com/Prathamesh0309/it-support-agent
cd it-support-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

## Usage

```bash
python phase1_ingest.py    # 1. build the vector store from data/
python phase1_rag.py       # 2. ask questions (RAG only)
python orchestrator.py     # 3. full agent: retrieval + tool actions
python evalute_rag.py      # 4. run the evaluation
```

## Evaluation

`evalute_rag.py` runs predefined test cases and scores generated answers against ground truth.

<!-- Add your results here: number of test cases and the score or pass rate. -->

## Notes and next steps

- Gemini is used for both embeddings and generation
- MCP tools are mocks; replace them with real ticketing and Slack integrations
- Next: richer chunking and metadata for retrieval, more knowledge sources in `data/`, and tests for tool invocation and end-to-end behavior
