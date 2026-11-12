# IT Support Agent

An agentic IT helpdesk assistant built on retrieval-augmented generation (RAG) and an MCP tool layer. It answers questions from a knowledge base, and when a question needs action it can open tickets and notify Slack through an MCP server.

## What it does

- **Retrieves** relevant support content from a local ChromaDB vector store using Gemini embeddings
- **Answers** with Gemini (`gemini-3.1-flash-lite`), grounded in the retrieved context; if nothing retrieved is relevant enough, it skips the LLM and escalates instead of guessing
- **Acts** through a FastMCP server exposing `create_ticket`, `check_ticket_status` and `notify_slack` (mock backends, designed to be swapped for real APIs)
- **Orchestrates** with a Plan → Act → Respond loop that decides whether to retrieve, call a tool, or answer
- **Filters tickets** in two layers: deterministic rules first, then an LLM-as-judge for ambiguous cases
- **Evaluates** retrieval and generation separately: hit rate and MRR for retrieval, an LLM judge (a different model from the generator) for faithfulness and answer relevancy

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
| `config.py` | Models, retrieval settings, relevance threshold, Gemini client with retry on 5xx |
| `evaluate_rag.py` | Scores retrieval and answers against ground-truth test cases (live API) |
| `test_mcp.py` | Smoke test: starts the MCP server and exercises all three tools |
| `tests/` | Offline unit tests (Gemini and tool calls mocked) |
| `data/` | Source knowledge documents (`.txt`) |
| `db/` | Local ChromaDB storage (generated) |
| `tickets.json` | Mock ticket store written by the MCP server (generated) |

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
python evaluate_rag.py      # 4. run the evaluation (calls the live API)
```

## Evaluation

`evaluate_rag.py` runs 7 test cases (6 in-scope questions and 1 out-of-scope question that should be refused) and reports:

| Metric | How it is measured |
| --- | --- |
| Retrieval hit rate and MRR | Did the expected source document appear in the top-K chunks? No LLM involved |
| Out-of-scope refusals | Did the relevance threshold stop the LLM from answering a question the knowledge base doesn't cover? |
| Faithfulness, answer relevancy | LLM-as-judge (`JUDGE_MODEL`, a different model from the generator) scoring 0 to 1 |

On my last run the judged scores were 0.95 to 1.0. Seven cases and an LLM judge give a rough signal, not a benchmark. The relevance threshold (`MIN_RELEVANCE = 0.62`) was calibrated on this small knowledge base, so re-run the evaluation whenever the documents or embedding model change.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite is offline and runs in CI: it covers the MCP tools, the relevance guardrail, and the orchestrator's routing rules (for example, a question the knowledge base can't answer always opens a ticket).

## Notes and next steps

- Gemini is used for embeddings, generation and (by default) judging; set `JUDGE_MODEL` in `.env` to change the judge
- MCP tools are mocks (tickets persist to `tickets.json`); replace them with real ticketing and Slack integrations
- `mcp` is pinned below 2.0 because 2.x renamed `FastMCP`
- Next: richer chunking and metadata for retrieval, more knowledge sources in `data/`, and end-to-end tests against a live model
