# Contract Intelligence System

An AI-fluency learning project that demonstrates **RAG, vector DBs, structured generation, guardrails, prompt engineering, agentic tool use, and LLM evaluation with Phoenix** — applied to contract review.

## What it does

Upload a contract → the system parses it, indexes it in a local vector DB, and then can:

- **Extract** structured clauses with Pydantic-validated schemas (guardrail).
- **Detect risks** with a hybrid rule-based + LLM scanner.
- **Answer questions** about the contract using RAG.
- **Summarize** with a structured prompt for a busy GC.
- **Run an agent** that decides which tools (search, extract, risks) to call.
- **Evaluate** answer quality with an LLM-as-judge harness, traced in Phoenix.

## AI fluency concepts on display

| Concept | Where it lives |
| --- | --- |
| Prompt engineering | `summarize.py`, `extraction.py`, `risk.py` (role + structure prompts) |
| Generative AI | All LLM calls via Replit AI (OpenAI proxy) |
| Structured output / Guardrails | `extraction.py`, `risk.py` (Pydantic + JSON-schema mode + retry on validation error) |
| RAG | `rag.py` + `vectorstore.py` |
| Vector DB | Chroma persistent client (`vectorstore.py`) |
| Embeddings | Chroma's local ONNX MiniLM (no API key, no cost) |
| Agentic AI | `agent.py` (OpenAI tool-calling loop) |
| Data pipeline | `pipeline.py` (file → parse → chunk → embed → metadata) |
| Evaluation | `eval.py` (LLM-judge correctness + groundedness) |
| Observability / tracing | `eval.py` (Phoenix auto-instrumentation) |

## Setup

Already installed in this Replit. The OpenAI integration is wired up — no API key needed.

## CLI

```bash
# Ingest the sample contract
python contract_intel/run_cli.py ingest contract_intel/sample_contracts/acme_msa.txt

# List indexed contracts
python contract_intel/run_cli.py list

# Structured clause extraction
python contract_intel/run_cli.py extract acme_msa

# Risk report
python contract_intel/run_cli.py risks acme_msa

# Executive summary
python contract_intel/run_cli.py summary acme_msa

# RAG question
python contract_intel/run_cli.py ask "What's the auto-renewal notice period?" --contract-id acme_msa

# Agent (decides which tools to use)
python contract_intel/run_cli.py review "Compare the liability caps and auto-renewal clauses across all my contracts."

# Run an eval set
python contract_intel/run_cli.py eval contract_intel/sample_contracts/sample_eval.json

# Launch Phoenix (free, local) for trace inspection
python contract_intel/run_cli.py phoenix
```

## API

```bash
python contract_intel/run_api.py
# then:
curl -F "file=@contract_intel/sample_contracts/acme_msa.txt" http://localhost:8001/contracts/upload
curl http://localhost:8001/contracts/acme_msa/extract
curl -X POST http://localhost:8001/ask -H "content-type: application/json" \
  -d '{"question":"What is the liability cap?","contract_id":"acme_msa"}'
```

## Project layout

```
contract_intel/
├── data/
│   ├── contracts/     # uploaded files + per-file .meta.json
│   └── chroma/        # persistent vector store
├── sample_contracts/
│   ├── acme_msa.txt
│   └── sample_eval.json
├── src/contract_intel/
│   ├── config.py       # paths, model defaults, env wiring
│   ├── llm.py          # thin OpenAI client wrapper (chat / chat_json / tools)
│   ├── ingestion.py    # PDF/DOCX/TXT → normalized chunks
│   ├── vectorstore.py  # Chroma persistent collection
│   ├── pipeline.py     # file → chunks → vector store + metadata sidecar
│   ├── extraction.py   # Pydantic schema + guardrail-style validate/retry
│   ├── risk.py         # rule-based + LLM risk detection
│   ├── rag.py          # retrieval-grounded Q&A with citations
│   ├── summarize.py    # GC-targeted exec summary prompt
│   ├── agent.py        # tool-using contract reviewer
│   ├── eval.py         # Phoenix tracing + LLM-judge eval harness
│   ├── cli.py          # Typer commands
│   └── api.py          # FastAPI app
├── run_cli.py
└── run_api.py
```

## Notes

- **Phoenix** is fully open-source and runs locally — no key, no cost.
- **Embeddings** use Chroma's bundled MiniLM model (CPU, local) — also free and zero-config.
- **LLM** calls go through Replit's AI Integrations OpenAI proxy.
- For DOCX/PDF support: `pypdf` and `python-docx` are already installed.
